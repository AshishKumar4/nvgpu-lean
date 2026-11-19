// The accelerated mode's device runtime: what a lifted kernel (tools/lift/lift.py) calls, each a
// transcription of the simulator's definition so the results are the simulator's bit for bit.
//
// - The tensor cores' datapath on words (NV.Num.TensorModel's `Datapath`, NV/Num/Tensor.lean) and
//   warpgroup MMAs on it (NV.Async.Wgmma: descriptors, swizzle, the accumulator fragment; the
//   Hopper bf16/f16 → f32 model), each thread computing its own accumulator values.
// - TMA tiled copies (NV.Async.Tma: `TensorMap.tile`), done by the issuing lane; tensor maps in the
//   simulator's encoding (NV.Async.TensorMap.encode), from the parameter block or global memory.
// - mbarriers in the simulator's word (NV.Async.MBarrierWord), by atomic read-modify-writes.
// - A CTA's shared memory is a window of device global memory (`Ctx`), so a Hopper CTA's 227 KB
//   fit on any GPU and cluster peers' windows are plain addresses.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// The launch's threads per CTA: lift.py --threads, from the launch the kernel is lifted for.
#ifndef LIFT_THREADS
#error "LIFT_THREADS: the launch's threads per CTA (lift.py --threads)"
#endif
#define LIFT_WARPS ((LIFT_THREADS + 31) / 32)

namespace lift {

struct Ctx {
  uint8_t* sharedAll;      // every CTA's shared window, `sharedBytes` apart
  uint8_t* shared;         // this CTA's
  uint64_t sharedBytes;
  const uint8_t* const0;   // constant bank 0 (device copy)
  int64_t globalDelta;     // device address = simulator address + globalDelta
  uint64_t hiBase;         // from here on (the emulated device's module data), + hiDelta instead
  int64_t hiDelta;
  uint32_t paramBase, paramBytes;
  uint32_t clusterRank, clusterBase;  // this CTA's rank, and the linear index of rank 0
  uint32_t clusterSize;
  uint32_t cx, cy, cz;     // the cluster's shape
  uint32_t gx, gy;         // the grid's x and y, for a peer's linear index
  unsigned long long* clusterBars;  // per cluster: `ClusterBar` (main.cu sets each to its live warps)
  uint8_t* tmemAll;        // every CTA's tensor memory (`NV.Async.Tcgen05`), TMEM_BYTES apart
  uint8_t* tmem;           // this CTA's
  uint32_t* fault;         // set nonzero on behaviour the model faults on
};

__device__ __forceinline__ uint64_t U64(uint32_t lo, uint32_t hi) { return (uint64_t)lo | ((uint64_t)hi << 32); }
__device__ __forceinline__ uint32_t ld32c(const Ctx& x, uint32_t off) { return *(const uint32_t*)(x.const0 + off); }
// A word of a kernel's own constant bank of `n` words (`Sem.resolveBanks`: misaligned or past the
// end reads 0).
#define CBANK(w, n, off) (((off) % 4u == 0u && (off) / 4u < (n)) ? (w)[(off) / 4u] : 0u)
__device__ __forceinline__ uint8_t* gptr(const Ctx& x, uint64_t a) {
  return (uint8_t*)(a + (uint64_t)(a >= x.hiBase ? x.hiDelta : x.globalDelta));
}

// The linear index of the CTA of cluster rank `rank`.
__device__ __forceinline__ uint64_t peerCta(const Ctx& x, uint32_t rank) {
  uint32_t bx = x.clusterBase % x.gx, by = (x.clusterBase / x.gx) % x.gy, bz = x.clusterBase / (x.gx * x.gy);
  uint32_t px = bx + rank % x.cx, py = by + (rank / x.cx) % x.cy, pz = bz + rank / (x.cx * x.cy);
  return px + (uint64_t)x.gx * (py + (uint64_t)x.gy * pz);
}

// `Exec.sharedAt`: a shared address `(rank << 24) | offset`: this CTA's window when the rank is
// its own, else the cluster peer's.
__device__ __forceinline__ uint8_t* sharedAt(const Ctx& x, uint32_t a) {
  uint32_t rank = a >> 24, off = a & 0xffffffu;
  if (rank == x.clusterRank) return x.shared + off;
  // The peer's CTA: rank 0's coordinates plus the rank's (x fastest, as the cluster rank counts).
  return x.sharedAll + peerCta(x, rank) * x.sharedBytes + off;
}

// Tensor memory (`NV.Async.Tcgen05`): 128 lanes of 512 four-byte columns, then the column
// allocator's word. The simulator's is per SM, shared by its CTAs (its allocator keeps them
// apart); here each CTA has its own, so columns a CTA allocates may differ from the simulator's
// where CTAs shared an SM, which no output depends on unless the kernel writes its column.
constexpr uint64_t TMEM_BYTES = 4ull * 128 * 512 + 256;
constexpr uint32_t TMEM_ALLOC = 4u * 128 * 512;
__device__ __forceinline__ uint32_t tmemByte(uint32_t lane, uint32_t col) { return 4u * (512u * lane + col); }
// A `.tmem` byte: this CTA's below 2^24, the cluster CTA of rank `(b >> 24) - 1`'s above
// (`Tcgen05.peerByte`).
__device__ __forceinline__ uint8_t* tmemAt(const Ctx& x, uint32_t b) {
  if (b < (1u << 24)) return x.tmem + b;
  return x.tmemAll + peerCta(x, (b >> 24) - 1) * TMEM_BYTES + (b & 0xffffffu);
}

// The cluster barrier (`UCGABAR_ARV`/`UCGABAR_WAIT`, the engine's `clusterBars`): per cluster, one
// word of arrived warps (bits 0-19), live warps (20-39) and the phase (40-63). A warp arrives once;
// the phase completes when the arrived reach the live (a warp's exit takes it out of the live).
__device__ inline uint32_t clusterStep(unsigned long long* w, bool arrive) {
  unsigned long long old = *(volatile unsigned long long*)w, assumed, nw;
  do {
    assumed = old;
    uint64_t arrived = assumed & 0xfffffull, live = (assumed >> 20) & 0xfffffull, phase = assumed >> 40;
    if (arrive) arrived += 1; else live -= 1;
    if (arrived > 0 && arrived >= live) { arrived = 0; phase += 1; }
    nw = arrived | (live << 20) | (phase << 40);
    old = atomicCAS(w, assumed, nw);
  } while (old != assumed);
  return (uint32_t)(assumed >> 40);
}

__device__ __forceinline__ unsigned long long* clusterBar(const Ctx& x) {
  uint32_t bx = x.clusterBase % x.gx, by = (x.clusterBase / x.gx) % x.gy, bz = x.clusterBase / (x.gx * x.gy);
  uint32_t ngx = x.gx / x.cx, ngy = x.gy / x.cy;
  return x.clusterBars + (bx / x.cx + ngx * (by / x.cy + ngy * (bz / x.cz)));
}

// `UCGABAR_ARV`: release, then the warp's arrival (its lowest executing lane's); returns the phase
// it arrived in, to every executing lane.
__device__ inline uint32_t clusterArrive(const Ctx& x) {
  __threadfence();
  uint32_t m = __activemask(), ph = 0;
  if (threadIdx.x % 32 == __ffs(m) - 1) ph = clusterStep(clusterBar(x), true);
  return __shfl_sync(m, ph, __ffs(m) - 1);
}

// `UCGABAR_WAIT`: until the phase arrived in has completed, then acquire.
__device__ inline void clusterWait(const Ctx& x, uint32_t phase) {
  while ((uint32_t)(*(volatile unsigned long long*)clusterBar(x) >> 40) == phase) {}
  __threadfence();
}

// A warp's exit: out of its cluster's live warps.
__device__ inline void clusterExit(const Ctx& x) {
  if (x.clusterSize > 1) clusterStep(clusterBar(x), false);
}

// ---------------------------------------------------------------------------------------------
// Integer helpers (NV/SASS/Exec.lean, Sem.lean).

__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {
  uint32_t r = 0;
#pragma unroll
  for (int k = 0; k < 8; ++k)
    if ((lut >> k) & 1u) r |= ((k & 4) ? a : ~a) & ((k & 2) ? b : ~b) & ((k & 1) ? c : ~c);
  return r;
}

__device__ __forceinline__ uint32_t shf(bool left, bool sgn, bool wide, bool hiWord, uint32_t lo, uint32_t hi, uint32_t s) {
  uint64_t x = U64(lo, hi);
  uint64_t bits = wide ? 64 : 32;
  uint64_t n = (uint64_t)s < bits ? (uint64_t)s : bits;
  uint64_t r;
  if (left) r = n >= 64 ? 0 : x << n;
  else if (sgn) r = (uint64_t)((int64_t)x >> (n < 63 ? n : 63));
  else r = n >= 64 ? 0 : x >> n;
  return hiWord ? (uint32_t)(r >> 32) : (uint32_t)r;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint64_t x = U64(a, b);
  uint32_t r = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t s = (sel >> (4 * k)) & 0xfu;
    uint32_t byte = (uint32_t)((x >> (8 * (s & 7u))) & 0xffu);
    if (s & 8u) byte = (byte & 0x80u) ? 0xffu : 0u;
    r |= byte << (8 * k);
  }
  return r;
}

// `cvt.rzi.u32.f32` (F2I.U32.TRUNC.NTZ): toward zero, clamped, NaN to 0.
__device__ __forceinline__ uint32_t f2u32_trunc(float f) { return __float2uint_rz(f); }

// ---------------------------------------------------------------------------------------------
// Binary32 with subnormals flushed (the `.FTZ` forms, NV.Num's `ftz`: inputs flushed, the result
// rounded with subnormals and then flushed). The lifted kernel is compiled with -ftz=false, so the
// `_rn` intrinsics round as IEEE does.

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// Packed pairs as their CUDA types and back.
__device__ __forceinline__ __half2 u2h(uint32_t x) { return *reinterpret_cast<__half2*>(&x); }
__device__ __forceinline__ __nv_bfloat162 u2bf(uint32_t x) { return *reinterpret_cast<__nv_bfloat162*>(&x); }
__device__ __forceinline__ uint32_t h2u(__half2 x) { return *reinterpret_cast<uint32_t*>(&x); }
__device__ __forceinline__ uint32_t bf2u(__nv_bfloat162 x) { return *reinterpret_cast<uint32_t*>(&x); }

__device__ __forceinline__ uint32_t flush32(uint32_t x) { return (x & 0x7f800000u) == 0 ? (x & 0x80000000u) : x; }
__device__ __forceinline__ float fl(uint32_t x) { return __uint_as_float(x); }
__device__ __forceinline__ uint32_t fadd32(bool ftz, uint32_t a, uint32_t b) {
  if (!ftz) return __float_as_uint(__fadd_rn(fl(a), fl(b)));
  return flush32(__float_as_uint(__fadd_rn(fl(flush32(a)), fl(flush32(b)))));
}
__device__ __forceinline__ uint32_t fmul32(bool ftz, uint32_t a, uint32_t b) {
  if (!ftz) return __float_as_uint(__fmul_rn(fl(a), fl(b)));
  return flush32(__float_as_uint(__fmul_rn(fl(flush32(a)), fl(flush32(b)))));
}
__device__ __forceinline__ uint32_t ffma32(bool ftz, uint32_t a, uint32_t b, uint32_t c) {
  if (!ftz) return __float_as_uint(__fmaf_rn(fl(a), fl(b), fl(c)));
  return flush32(__float_as_uint(__fmaf_rn(fl(flush32(a)), fl(flush32(b)), fl(flush32(c)))));
}

// ---------------------------------------------------------------------------------------------
// The special-function unit (NV.Num.MufuWord, NV.Num.Mufu): Carleton's interpolation with the
// tables of NV/Num/MufuTables.lean (mufu_tables.cuh, generated from it by lift.py).

#include "mufu_tables.cuh"

__device__ __forceinline__ uint64_t mufuSquare(uint64_t w) {
  uint64_t diag = 0;
  for (int i = 0; i < 16; ++i) if ((w >> i) & 1) diag += 1ull << (2 * i);
  const uint64_t m19 = 0x7ffff;
  uint64_t err = diag & m19;
  for (int i = 0; i < 9; ++i)
    if ((w >> i) & 1) {
      uint64_t row = (w << (i + 1)) & 0xffffffffull;
      err += (row - (row & ((1ull << (2 * i + 2)) - 1))) & m19;
    }
  uint64_t sq = w * w;
  return err <= sq ? sq - err : 0;
}

struct MufuPoly { const uint32_t (*t)[3]; int a0, a1; bool neg1, neg2; int weight; uint64_t bias; };

__device__ __forceinline__ uint64_t mufuEval(const MufuPoly& p, int index, uint64_t xl, uint64_t sq) {
  uint64_t c0 = p.t[index][0], c1 = p.t[index][1], c2 = p.t[index][2];
  uint64_t t1 = p.a1 < 64 ? (c1 * xl) << p.a1 : 0;
  uint64_t t2 = c2 * sq;
  uint64_t s = (p.a0 < 64 ? c0 << p.a0 : 0) + (p.neg1 ? 0 - t1 : t1) + (p.neg2 ? 0 - t2 : t2);
  return s + (p.bias >> (64 - p.weight + 23));
}
__device__ __forceinline__ uint32_t mufuFrac(const MufuPoly& p, uint64_t sum) {
  return (uint32_t)((sum >> (p.weight - 23)) & 0x7fffff);
}

#define MUFU_EX2 MufuPoly{kEx2, 32, 19, false, false, 57, 0x6fc4000000000000ull}
#define MUFU_LG2 MufuPoly{kLg2, 30, 18, false, true, 56, 0x868b000000000000ull}
#define MUFU_RCP MufuPoly{kRcp, 30, 17, true, false, 56, 0x67e7000000000000ull}
#define MUFU_RSQ MufuPoly{kRsqrt, 31, 17, true, false, 57, 0x7fff800000000000ull}

// `x · 2^k` for a normal `x`, rounded to nearest with subnormals, then flushed (MufuWord.ldexp):
// exact in binary64, one rounding to binary32.
__device__ __forceinline__ uint32_t mufuLdexp(uint32_t x, int k) {
  int ef = (x >> 23) & 0xff;
  double m = (double)((x & 0x7fffff) | 0x800000);
  double v = ldexp(m, ef - 150 + k);
  uint32_t r = __float_as_uint(__double2float_rn(v));
  return flush32(r) | (x & 0x80000000u);
}

__device__ __forceinline__ bool isNaN32(uint32_t x) { return (x & 0x7fffffffu) > 0x7f800000u; }

// `MUFU.EX2` (MufuWord.ex2).
__device__ inline uint32_t mufuEx2(uint32_t x) {
  bool neg = x >> 31;
  uint32_t ef = (x >> 23) & 0xff;
  if (isNaN32(x)) return 0x7fffffffu;
  if (ef == 0xff) return neg ? 0u : 0x7f800000u;
  if (ef == 0) return 0x3f800000u;
  if (ef >= 150) return neg ? 0u : 0x7f800000u;
  uint64_t m = (x & 0x7fffff) | 0x800000;
  uint64_t k = 150 - ef;
  uint64_t ip, fp;
  if (k < 24) { ip = m >> k; fp = m & ((1ull << k) - 1); } else { ip = 0; fp = m; }
  uint64_t xb = k <= 23 ? fp << (23 - k) : (k < 47 ? fp >> (k - 23) : 0);
  int64_t i;
  if (neg && xb != 0) { xb = 0xffffffffull - xb; i = -(int64_t)ip - 1; } else { i = neg ? -(int64_t)ip : (int64_t)ip; }
  xb &= 0x7fffff;
  const MufuPoly P = MUFU_EX2;
  uint64_t xl = xb & 0x1ffff;
  uint64_t sum = mufuEval(P, (int)(xb >> 17), xl, mufuSquare(xl));
  uint32_t scale = i > 127 ? 0x7f800000u : (i < -126 ? 0u : (uint32_t)(i + 127) << 23);
  return fmul32(true, scale, 0x3f800000u | mufuFrac(P, sum));
}

// `MUFU.RCP` (MufuWord.rcp).
__device__ inline uint32_t mufuRcp(uint32_t x) {
  x = flush32(x);
  bool neg = x >> 31;
  uint32_t ef = (x >> 23) & 0xff;
  if (isNaN32(x)) return 0x7fffffffu;
  if (ef == 0xff) return neg ? 0x80000000u : 0u;
  if (ef == 0) return neg ? 0xff800000u : 0x7f800000u;
  const MufuPoly P = MUFU_RCP;
  uint64_t xl = x & 0xffff;
  uint64_t sum = mufuEval(P, (int)((x >> 16) & 0x7f), xl, mufuSquare(2 * xl) / 4);
  uint32_t re = 127;
  if ((sum >> 56) == 0) { re = 126; sum = 2 * sum; }
  return mufuLdexp((neg ? 0x80000000u : 0u) | (re << 23) | mufuFrac(P, sum), 127 - (int)ef);
}

// `MUFU.RSQ` (MufuWord.rsqrt).
__device__ inline uint32_t mufuRsq(uint32_t x) {
  x = flush32(x);
  bool neg = x >> 31;
  uint32_t ef = (x >> 23) & 0xff;
  if (isNaN32(x)) return 0x7fffffffu;
  if (ef == 0xff) return neg ? 0x7fffffffu : 0u;
  if (ef == 0) return neg ? 0xff800000u : 0x7f800000u;
  if (neg) return 0x7fffffffu;
  int l = (int)ef - 127;
  uint32_t fr = x & 0x7fffff;
  if (l % 2 == 0 && fr == 0) return mufuLdexp(0x3f800000u, -(l / 2));
  int index = fr >> 17;
  uint64_t xl = fr & 0x1ffff;
  if (l % 2 != 0) { index += 64; l -= 1; }
  const MufuPoly P = MUFU_RSQ;
  uint64_t sum = 2 * mufuEval(P, index, xl, mufuSquare(xl));
  return mufuLdexp((126u << 23) | mufuFrac(P, sum), -(l / 2));
}

// `MUFU.LG2` (Mufu.lg2 on a flushed input).
__device__ inline uint32_t mufuLg2(uint32_t x) {
  x = flush32(x);
  bool neg = x >> 31;
  uint32_t ef = (x >> 23) & 0xff;
  if (isNaN32(x)) return 0x7fffffffu;
  if (ef == 0xff) return neg ? 0x7fffffffu : 0x7f800000u;
  if (ef == 0) return 0xff800000u;
  if (neg) return 0x7fffffffu;
  if (x == 0x3f800000u) return 0u;
  const MufuPoly P = MUFU_LG2;
  uint32_t fr = x & 0x7fffff;
  uint64_t xl = fr & 0x1ffff;
  __int128 sum = (__int128)mufuEval(P, (int)(fr >> 17), xl, mufuSquare(xl)) + (__int128)((int)ef - 127) * ((__int128)1 << 56);
  unsigned __int128 v = sum < 0 ? (unsigned __int128)(-sum) : (unsigned __int128)sum;
  v = v >> 20 << 20;
  int re = 127;
  for (int it = 0; it < 64; ++it) {
    if ((v >> 56) == 0 && v != 0) { re -= 1; v = 2 * v; }
    else if ((v >> 57) != 0) { re += 1; v = v / 2; }
    else break;
  }
  bool small = (x & 0x7fffffffu) < 0x3f800000u;
  return (small ? 0x80000000u : 0u) | ((uint32_t)re << 23) | mufuFrac(P, (uint64_t)v);
}

// ---------------------------------------------------------------------------------------------
// The tensor-core datapath (NV.Num.TensorModel.Datapath), for operands of up to 11 fraction bits.

struct Codec { uint64_t pad, man, ex, bias; bool sign, ieee, nanOnly, sub; };

__device__ __forceinline__ Codec codecIeee(int exp, int man) {
  Codec c; c.pad = 0; c.man = man; c.ex = exp; c.bias = (1u << (exp - 1)) - 1;
  c.sign = true; c.ieee = true; c.nanOnly = false; c.sub = true; return c;
}

// `Codec.encode`: significand in bits 0-11, units exponent + 512 in 12-21, sign in 22, class in
// 23-24 (1 infinity, 2 NaN, 3 zero).
__device__ __forceinline__ uint32_t encode(const Codec& c, uint32_t x) {
  uint64_t w = (uint64_t)x >> c.pad;
  uint64_t frac = w & ((1ull << c.man) - 1);
  uint64_t ef = (w >> c.man) & ((1ull << c.ex) - 1);
  uint32_t sign = (c.sign && ((w >> (c.man + c.ex)) & 1)) ? 0x400000u : 0u;
  bool top = ef == (1ull << c.ex) - 1;
  if (top && c.ieee) return frac == 0 ? (0x800000u | sign) : 0x1000000u;
  if (top && c.nanOnly && frac == (1ull << c.man) - 1) return 0x1000000u;
  if (ef == 0 && c.sub) {
    if (frac == 0) return 0x1800000u | sign;
    return sign | ((uint32_t)(513 - c.bias) << 12) | (uint32_t)frac;
  }
  return sign | ((uint32_t)(ef + 512 - c.bias) << 12) | (uint32_t)(frac | (1ull << c.man));
}

struct Datapath {
  bool shl; uint64_t sh; int64_t alignFrac, minAlignExp; bool rn;
  uint64_t cMan, cExp, cPad; int64_t cBias, cQMin; uint64_t cExpMax;
  uint64_t oMan, oPad; int64_t oQMin; uint64_t signBit, nan;
};

// `TensorModel.prepare` with an IEEE binary32 accumulator (`outF = f32`).
__device__ __forceinline__ Datapath prepareF32(int alignFrac, int minAlignExp, int outFrac, bool rn, int aMan, int bMan) {
  Datapath d;
  int sl = alignFrac - aMan - bMan;
  d.shl = sl >= 0; d.sh = sl >= 0 ? sl : -sl;
  d.alignFrac = alignFrac; d.minAlignExp = minAlignExp; d.rn = rn;
  d.cMan = 23; d.cExp = 8; d.cPad = 0; d.cBias = 127; d.cQMin = 1 - 127 - 23; d.cExpMax = 255;
  d.oMan = outFrac; d.oPad = 23 - outFrac; d.oQMin = 1 - 127 - outFrac;
  d.signBit = 1ull << 31; d.nan = 0x7fffffffull;
  return d;
}

__device__ __forceinline__ uint32_t expSum(uint32_t x, uint32_t y) { return ((x >> 12) & 0x3ffu) + ((y >> 12) & 0x3ffu); }
__device__ __forceinline__ int64_t log2Word(uint64_t x) { return 63 - __clzll((long long)x); }

// `Datapath.round`.
__device__ inline uint64_t roundOut(const Datapath& d, bool neg, uint64_t M, int64_t E) {
  const uint64_t one = 1;
  uint64_t sign = neg ? d.signBit : 0;
  uint64_t man = d.oMan;
  int64_t q0 = E + log2Word(M) - (int64_t)man;
  int64_t q = q0 < d.oQMin ? d.oQMin : q0;
  int64_t k = q - E;
  uint64_t t;
  if (k <= 0) t = M << (uint64_t)(-k);
  else if (k >= 64) t = 0;
  else {
    t = M >> (uint64_t)k;
    uint64_t rem = M & ((one << (uint64_t)k) - 1);
    uint64_t half = one << (uint64_t)(k - 1);
    if (d.rn && (rem > half || (rem == half && (t & 1)))) t = t + 1;
  }
  if (t == (one << (man + 1))) { t = one << man; q = q + 1; }
  if (t == 0) return sign;
  uint64_t ef = t < (one << man) ? 0 : (uint64_t)(q - d.oQMin + 1);
  if (ef >= d.cExpMax) ef = d.cExpMax;
  uint64_t frac = ef == d.cExpMax ? 0 : (t & ((one << man) - 1));
  return sign | (((ef << man) | frac) << d.oPad);
}

// `Datapath.blockAt` over all `n` products: `Σ a[i]·b[i] + c`.
__device__ inline uint64_t blockAt(const Datapath& d, const uint32_t* a, const uint32_t* b, int n, uint64_t c) {
  uint32_t emax = (uint32_t)(d.minAlignExp + 1024);
  uint32_t flags = 0;
  for (int i = 0; i < n; ++i) {
    uint32_t x = a[i], y = b[i];
    if (((x | y) >> 23) == 0) {
      uint32_t e = expSum(x, y);
      if (e > emax) emax = e;
      flags |= 8;
    } else {
      uint32_t cx = x >> 23, cy = y >> 23;
      if (cx == 2 || cy == 2 || (cx == 3 && cy == 1) || (cx == 1 && cy == 3)) flags |= 1;
      else if (cx == 3 || cy == 3) {}
      else if (((x ^ y) >> 22) & 1) flags |= 4;
      else flags |= 2;
    }
  }
  uint64_t w = c >> d.cPad;
  uint64_t frac = w & ((1ull << d.cMan) - 1);
  uint64_t ef = (w >> d.cMan) & ((1ull << d.cExp) - 1);
  bool cneg = (w >> (d.cMan + d.cExp)) & 1;
  bool cspecial = ef == d.cExpMax;
  bool cnonzero = !cspecial && (ef != 0 || frac != 0);
  flags |= cspecial ? (frac != 0 ? 1u : (cneg ? 4u : 2u)) : (cnonzero ? 8u : 0u);
  if ((flags & 1) || (flags & 6) == 6) return d.nan;
  if (flags & 2) return d.cExpMax << (d.cMan + d.cPad);
  if (flags & 4) return d.signBit | (d.cExpMax << (d.cMan + d.cPad));
  if (!(flags & 8)) return 0;
  uint64_t m = ef == 0 ? frac : (frac | (1ull << d.cMan));
  int64_t e = ef == 0 ? d.cQMin : (int64_t)ef - d.cBias - (int64_t)d.cMan;
  int64_t ce0 = e + (m ? log2Word(m) : 0);
  int64_t ce = !cnonzero ? d.minAlignExp : (ce0 < -126 ? -126 : ce0);
  int64_t k = e - ce + d.alignFrac;
  uint64_t cm = !cnonzero ? 0 : (k >= 0 ? m << (uint64_t)k : m >> (uint64_t)(-k));
  uint32_t cf = (uint32_t)(ce + 1024);
  if (cf > emax) emax = cf;
  int64_t s = 0;
  for (int i = 0; i < n; ++i) {
    uint32_t x = a[i], y = b[i];
    if (((x | y) >> 23) == 0) {
      uint64_t sx = x & 0xfffu, sy = y & 0xfffu;
      uint64_t p = d.shl ? (sx * sy) << d.sh : (sx * sy) >> d.sh;
      uint32_t dd = emax - expSum(x, y);
      int64_t v = dd >= 64 ? 0 : (int64_t)(p >> dd);
      s = (((x ^ y) >> 22) & 1) ? s - v : s + v;
    }
  }
  uint32_t dc = emax - cf;
  int64_t cv = dc >= 64 ? 0 : (int64_t)(cm >> dc);
  s = cneg ? s - cv : s + cv;
  if (s == 0) return 0;
  return roundOut(d, s < 0, s < 0 ? (uint64_t)(-s) : (uint64_t)s, (int64_t)emax - 1024 - d.alignFrac);
}

// ---------------------------------------------------------------------------------------------
// Warpgroup MMA (NV.Async.Wgmma), 16-bit inputs from shared memory, f32 accumulator.

enum InType { F16 = 0, BF16 = 1 };

// `Swizzle.apply` for wgmma's descriptor modes (`Desc.decode`: 1 b128, 2 b64, 3 b32).
__device__ __forceinline__ uint32_t descSwizzle(int mode, uint32_t a) {
  uint32_t line = a >> 7;
  switch (mode) {
    case 1: return a ^ ((line & 7u) << 4);
    case 2: return a ^ ((line & 3u) << 4);
    case 3: return a ^ ((line & 1u) << 4);
    default: return a;
  }
}

// `Desc.addr`: the shared address of element (mn, k) of a 2-byte operand.
__device__ __forceinline__ uint32_t descAddr(uint64_t desc, bool kMajor, int mn, int k, uint32_t* fault) {
  uint32_t start = (uint32_t)(desc & 0x3fff) << 4;
  uint32_t lbo = (uint32_t)((desc >> 16) & 0x3fff) << 4;
  uint32_t sbo = (uint32_t)((desc >> 32) & 0x3fff) << 4;
  if ((desc >> 49) & 7) atomicOr(fault, 2u);  // a non-zero matrix base offset is not modelled
  int mode = (int)((desc >> 62) & 3);
  uint32_t W = mode == 0 ? 0 : mode == 1 ? 128 : mode == 2 ? 64 : 32;
  const uint32_t esz = 2, T = 8;
  uint32_t off;
  if (kMajor) {
    if (W == 0) off = (mn % 8) * 16 + (mn / 8) * sbo + (k % T) * esz + (k / T) * lbo;
    else off = (mn % 8) * W + (mn / 8) * sbo + k * esz;
  } else {
    if (W == 0) off = (mn % T) * esz + (mn / T) * sbo + (k % 8) * 16 + (k / 8) * lbo;
    else {
      uint32_t w = W / 16;
      off = (mn % T) * esz + ((mn / T) % w) * 16 + (mn / (T * w)) * lbo + (k % 8) * W + (k / 8) * sbo;
    }
  }
  return descSwizzle(mode, start + off);
}

// One thread's share of `HGMMA.64xNx16.F32[.BF16]` from shared memory: its `N/2` accumulator
// words `acc`, updated in place (`scaleD` false: from zero). Thread `t` of the warpgroup holds
// row `16·(t/32) + (t%32)/4 + 8·((v/2) mod 2)`, column `2·(t%4) + v mod 2 + 8·(v/4)` of its value v
// (`accPos`). Hopper's datapath for f16/bf16 inputs and an f32 accumulator: blocks of 16, 25
// fraction bits aligned, exponent floor -133, the sum truncated to binary32 (`Hopper.f16f32`).
template <int N>
__device__ inline void hgmma16(const Ctx& x, InType ty, uint64_t da, uint64_t db, bool aKMajor, bool bKMajor,
                               bool negA, bool negB, bool scaleD, uint32_t* acc, uint32_t tid) {
  const int man = ty == BF16 ? 7 : 10;
  const Codec c = ty == BF16 ? codecIeee(8, 7) : codecIeee(5, 10);
  const Datapath dp = prepareF32(25, -133, 23, false, man, man);
  const int w = (tid / 32) % 4, l = tid % 32;
  uint32_t ea[2][16];
  for (int h = 0; h < 2; ++h) {
    int row = 16 * w + l / 4 + 8 * h;
    for (int k = 0; k < 16; ++k) {
      uint16_t v = __ldca((const unsigned short*)(x.shared + descAddr(da, aKMajor, row, k, x.fault)));
      if (negA) v ^= 0x8000;
      ea[h][k] = encode(c, v);
    }
  }
  for (int cb = 0; cb < N / 8; ++cb) {
    for (int j = 0; j < 2; ++j) {
      int col = 8 * cb + 2 * (l % 4) + j;
      uint32_t eb[16];
      for (int k = 0; k < 16; ++k) {
        uint16_t u = __ldca((const unsigned short*)(x.shared + descAddr(db, bKMajor, col, k, x.fault)));
        if (negB) u ^= 0x8000;
        eb[k] = encode(c, u);
      }
      for (int h = 0; h < 2; ++h) {
        int v = 4 * cb + 2 * h + j;
        acc[v] = (uint32_t)blockAt(dp, ea[h], eb, 16, scaleD ? acc[v] : 0u);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// 5th-generation tensor cores (`NV.Async.Tcgen05`, `NV.SASS.Async`).

// `Tcgen05.rowLane`: the lane of row `i` of an M-row operand from lane `l0`.
__device__ __forceinline__ uint32_t rowLane(uint32_t m, uint32_t l0, uint32_t i) {
  return m == 128 ? l0 + i : l0 + 32 * (i / 16) + i % 16;
}

// `UTCHMMA[.2CTA]`: one `tcgen05.mma` of `.kind::f16` (f16 or bf16 inputs into f32), A and B from
// shared memory, run by one thread when it issues (`Tcgen05.Inst.op`): D = A·B (+ D), the B200's
// datapath (`NV.Num.Blackwell.f16f32`: Hopper's). `pair`: `.cta_group::2`, M 256, rows and
// columns 128/N2 on the peer (the next rank). Others set the fault flag.
__device__ inline void utcmma(const Ctx& x, uint64_t da, uint64_t db, uint32_t d, uint32_t idesc, bool accumulate,
                              bool pair) {
  auto f = [&](int lo, int w) { return (idesc >> lo) & ((1u << w) - 1); };
  uint32_t dtype = f(4, 2), atype = f(7, 3), btype = f(10, 3);
  bool negA = f(13, 1), negB = f(14, 1), transA = f(15, 1), transB = f(16, 1);
  uint32_t N = f(17, 6) << 3, M = f(24, 5) << 4;
  bool bad = f(2, 1) || f(30, 2) || dtype != 1 || atype > 1 || btype != atype ||
             (pair ? M != 256 : (M != 64 && M != 128)) || N < 8 || N > 256 || ((da >> 61) & 1) || ((db >> 61) & 1) ||
             ((da >> 46) & 7) != 1 || ((db >> 46) & 7) != 1;
  uint32_t dl = d >> 16, dc = d & 0xffffu;
  if (bad || dc + N > 512 || (M == 64 ? (dl != 0 && dl != 16) : dl != 0)) { atomicOr(x.fault, 32u); return; }
  InType ty = atype == 1 ? BF16 : F16;
  const int man = ty == BF16 ? 7 : 10;
  const Codec c = ty == BF16 ? codecIeee(8, 7) : codecIeee(5, 10);
  const Datapath dp = prepareF32(25, -133, 23, false, man, man);
  uint32_t peer = x.clusterRank + 1;
  // A row's or B column's shared memory: this CTA's for the first half of a pair, the peer's after.
  auto elem = [&](uint64_t desc, bool kMajor, uint32_t n, uint32_t half, uint32_t mn, int k, bool neg) {
    uint32_t r = mn, rank = x.clusterRank;
    if (pair && mn >= half) { r = mn - half; rank = peer; }
    uint32_t a = descAddr(desc, kMajor, r, k, x.fault);
    uint16_t v = *(volatile uint16_t*)sharedAt(x, (rank << 24) | a);
    if (neg) v ^= 0x8000;
    return encode(c, v);
  };
  for (uint32_t i = 0; i < M; ++i) {
    uint32_t ea[16];
    for (int k = 0; k < 16; ++k) ea[k] = elem(da, !transA, M, 128, i, k, negA);
    for (uint32_t j = 0; j < N; ++j) {
      uint32_t eb[16];
      for (int k = 0; k < 16; ++k) eb[k] = elem(db, !transB, N, N / 2, j, k, negB);
      uint32_t b = pair && i >= 128 ? ((peer + 1) << 24) + tmemByte(i - 128, dc + j)
                                    : tmemByte(pair ? i : rowLane(M, dl, i), dc + j);
      uint32_t* w = (uint32_t*)tmemAt(x, b);
      *w = (uint32_t)blockAt(dp, ea, eb, 16, accumulate ? *w : 0u);
    }
  }
  __threadfence();
}

// `UTCATOMSWS.FIND_AND_SET.ALIGN`: claim `n` granules of 32 columns at a multiple of `n` in the
// allocator word (`findSetAlign`); the first granule, or -1.
__device__ inline int tmemAlloc(const Ctx& x, uint32_t n) {
  unsigned* w = (unsigned*)(x.tmem + TMEM_ALLOC);
  if (n == 0 || n > 32 || (n & (n - 1))) { atomicOr(x.fault, 32u); return -1; }
  unsigned old = *(volatile unsigned*)w, assumed;
  do {
    assumed = old;
    int g = -1;
    for (uint32_t k = 0; k < 32 / n; ++k)
      if (((uint64_t)assumed >> (k * n)) % (1ull << n) == 0) { g = (int)(k * n); break; }
    if (g < 0) return -1;
    old = atomicCAS(w, assumed, assumed | (unsigned)((((uint64_t)1 << n) - 1) << g));
    if (old == assumed) return g;
  } while (true);
}

// `UTCATOMSWS.AND`: the allocator word ANDed with `mask`; the old word.
__device__ inline uint32_t tmemFree(const Ctx& x, uint32_t mask) {
  return atomicAnd((unsigned*)(x.tmem + TMEM_ALLOC), mask);
}

// `Tcgen05.Shape.pos`: the lane and column of register `i` of thread `t` (shapes: 0 32x32b,
// 1 16x64b, 2 16x128b, 3 16x256b, 4 16x32bx2).
__device__ __forceinline__ void tmemPos(int shape, uint32_t t, uint32_t i, uint32_t l0, uint32_t c0,
                                        uint32_t* lane, uint32_t* col) {
  switch (shape) {
    case 0: *lane = l0 + t; *col = c0 + i; break;
    case 1: *lane = l0 + t / 4 + 8 * (t % 2); *col = c0 + 2 * i + (t / 2) % 2; break;
    case 2: *lane = l0 + t / 4 + 8 * (i % 2); *col = c0 + 4 * (i / 2) + t % 4; break;
    case 3: *lane = l0 + t / 4 + 8 * ((i % 4) / 2); *col = c0 + 8 * (i / 4) + 2 * (t % 4) + i % 2; break;
    default: *lane = l0 + t; *col = c0 + i; break;
  }
}

// `LDTM`/`STTM` for this thread: `regs` registers at `taddr` (`moveReq`); `pack`, two 16-bit
// columns a register. The lanes must be the warp's quarter; an address in lanes 0-31 names it.
__device__ inline void tmemMove(const Ctx& x, bool ld, int shape, uint32_t regs, bool pack, uint32_t taddr,
                                uint32_t warp, uint32_t t, uint32_t* r) {
  uint32_t lo = 32 * (warp % 4), a = taddr >> 16, c0 = taddr & 0xffffu, l0 = a < 32 ? lo + a : a;
  uint32_t per = pack ? 2 * regs : regs;
  for (uint32_t j = 0; j < per; ++j) {
    uint32_t lane, col;
    tmemPos(shape, t, j, l0, c0, &lane, &col);
    if (lane < lo || lane >= lo + 32 || col >= 512) { atomicOr(x.fault, 32u); return; }
    uint32_t* w = (uint32_t*)(x.tmem + tmemByte(lane, col));
    if (pack) {
      uint32_t& reg = r[j / 2];
      int sh = 16 * (j % 2);
      if (ld) reg = (reg & ~(0xffffu << sh)) | ((*w & 0xffffu) << sh);
      else *w = (*w & 0xffff0000u) | ((reg >> sh) & 0xffffu);
    } else {
      if (ld) r[j] = *w; else *w = r[j];
    }
  }
}

// ---------------------------------------------------------------------------------------------
// mbarriers in the simulator's word (NV.Async.MBarrierWord): bits 0-19 pending arrivals, 20-39
// expected, 40-60 tx-count (21-bit two's complement), 63 the phase parity.

__device__ __forceinline__ uint64_t mbField(uint64_t w, int lo, int width) { return (w >> lo) & ((1ull << width) - 1); }

// `MBarrierWord.apply`: `kind` 0 init(n), 1 arrive(n) with expect-tx(tx), 2 complete-tx(tx).
__device__ inline uint64_t mbApply(uint64_t old, int kind, uint32_t n, int64_t tx, bool drop, uint32_t* fault) {
  uint64_t phase = mbField(old, 63, 1), pending = mbField(old, 0, 20), expected = mbField(old, 20, 20);
  int64_t t = (int64_t)mbField(old, 40, 21);
  if (t >= (1 << 20)) t -= (1 << 21);
  const int64_t cmax = (1 << 20) - 1;
  bool bad = false;
  if (kind == 0) {
    if (n < 1 || n > (uint32_t)cmax) bad = true;
    phase = 0; pending = n; expected = n; t = 0;
  } else if (kind == 1) {
    t += tx; if (t < -cmax || t > cmax) bad = true;
    if (drop) expected -= n;
    if (n) {
      if (n > pending) bad = true;
      else { pending -= n; if (pending == 0 && t == 0) { phase ^= 1; pending = expected; } }
    }
  } else {
    t -= tx; if (t < -cmax || t > cmax) bad = true;
    if (pending == 0 && t == 0) { phase ^= 1; pending = expected; }
  }
  if (bad) atomicOr(fault, 1u);
  uint64_t tw = (uint64_t)(t < 0 ? t + (1 << 21) : t);
  return (pending & 0xfffffull) | ((expected & 0xfffffull) << 20) | ((tw & 0x1fffffull) << 40) | (phase << 63);
}

__device__ inline uint64_t mbAtomic(unsigned long long* w, int kind, uint32_t n, int64_t tx, bool drop, uint32_t* fault) {
  unsigned long long old = *(volatile unsigned long long*)w, assumed;
  do {
    assumed = old;
    old = atomicCAS(w, assumed, (unsigned long long)mbApply(assumed, kind, n, tx, drop, fault));
  } while (old != assumed);
  return assumed;
}

__device__ __forceinline__ unsigned long long* MB(const Ctx& x, uint32_t a) { return (unsigned long long*)sharedAt(x, a); }

// `SYNCS.EXCH.64`: initialize an mbarrier from its encoded arrival count (`initCount?`).
__device__ inline void mbExch(const Ctx& x, uint32_t a, uint64_t v) {
  uint64_t c = (v >> 1) & ((1ull << 20) - 1);
  uint64_t n = (1ull << 20) - c;
  uint64_t cc = (1ull << 20) - n;
  if (!(c >= 1 && ((cc << 1) | (cc << 43)) == v)) { atomicOr(x.fault, 4u); return; }
  __threadfence();
  atomicExch(MB(x, a), (unsigned long long)mbApply(0, 0, (uint32_t)n, 0, false, x.fault));
}

// ---------------------------------------------------------------------------------------------
// TMA tiled copies (NV.Async.Tma) with tensor maps in NVIDIA's descriptor layout
// (NV.Async.TensorMap.encode).

struct TensorMapView {
  uint64_t addr; uint64_t dims[5]; uint64_t strides[4]; uint32_t box[5]; uint32_t es[5];
  int rank; int esz; int swizzle; bool ok;
};

__device__ __forceinline__ uint64_t leAt(const uint8_t* p, int off, int bytes) {
  uint64_t v = 0;
  for (int i = bytes - 1; i >= 0; --i) v = (v << 8) | p[off + i];
  return v;
}

// `w` bits (at most 32) of the descriptor from bit `at`.
__device__ __forceinline__ uint32_t mapBits(const uint8_t* p, int at, int w) {
  return (uint32_t)((leAt(p, at / 8, 8) >> (at % 8)) & ((1ull << w) - 1));
}

// `TensorMap.decode` for a tiled map with a 16-byte swizzle atomicity: the address, rank − 1 (bits
// 68–70), element type (71–74, tf32 flag 80), interleave (75–76), swizzle mode (77–78), strides in
// 16-byte units (96 on, high bits 224 on), dimensions − 1 (256 on), element strides − 1 (416 on),
// box dimension 0 − 1 (440) and 1–4 (448 on); an im2col map (bit 64) is refused.
__device__ inline TensorMapView decodeMap(const uint8_t* p) {
  TensorMapView m;
  // Element bytes by code: u8 u16 u32 s32 u64 s64 f16 f32 f32.ftz f64 bf16.
  const int esz[] = {1, 2, 4, 4, 8, 8, 2, 4, 4, 8, 2, 0, 0, 0, 0, 0};
  uint32_t dt = mapBits(p, 71, 4);
  m.ok = mapBits(p, 75, 2) == 0 && mapBits(p, 83, 4) == 0 && mapBits(p, 64, 1) == 0;
  m.rank = (int)mapBits(p, 68, 3) + 1; m.esz = esz[dt]; m.swizzle = (int)mapBits(p, 77, 2);
  m.addr = leAt(p, 0, 8);
  for (int i = 0; i < 5; ++i) m.dims[i] = (uint64_t)mapBits(p, 256 + 32 * i, 32) + 1;
  for (int i = 0; i < 4; ++i)
    m.strides[i] = 16 * ((uint64_t)mapBits(p, 96 + 32 * i, 32) + ((uint64_t)mapBits(p, 224 + 4 * i, 4) << 32));
  m.box[0] = mapBits(p, 440, 8) + 1;
  for (int i = 1; i < 5; ++i) m.box[i] = mapBits(p, 448 + 8 * (i - 1), 8) + 1;
  for (int i = 0; i < 5; ++i) m.es[i] = mapBits(p, 416 + 3 * i, 3) + 1;
  if (m.es[0] != 1 || m.esz == 0 || m.rank > 5) m.ok = false;
  return m;
}

// `paramMap?`: an offset within the parameter block names a parameter's map, else a global address.
__device__ inline TensorMapView mapAt(const Ctx& x, uint64_t a) {
  const uint8_t* p = a + 128 <= x.paramBytes ? x.const0 + x.paramBase + a : gptr(x, a);
  TensorMapView m = decodeMap(p);
  if (!m.ok) atomicOr(x.fault, 8u);
  return m;
}

// `Swizzle.apply` for a tensor map's modes (none, 32B, 64B, 128B).
__device__ __forceinline__ uint32_t tmaSwizzle(int mode, uint32_t a) {
  uint32_t line = a >> 7;
  switch (mode) {
    case 1: return a ^ ((line & 1u) << 4);
    case 2: return a ^ ((line & 3u) << 4);
    case 3: return a ^ ((line & 7u) << 4);
    default: return a;
  }
}

// The box at `coords`, element by element: `f(global address or 0 when outside, shared offset)`
// with the shared offsets swizzled from `smemOff` (`TensorMap.tile`). Returns the box's bytes.
template <class F>
__device__ inline uint32_t tile(const TensorMapView& m, const int32_t* coords, uint32_t smemOff, F f) {
  int r = m.rank;
  uint32_t cnt[5];
  for (int i = 0; i < 5; ++i) cnt[i] = i < r ? (m.box[i] + m.es[i] - 1) / (m.es[i] ? m.es[i] : 1) : 1;
  uint32_t rowBytes = cnt[0] * m.esz;
  uint32_t rows = 1;
  for (int i = 1; i < r; ++i) rows *= cnt[i];
  for (uint32_t row = 0; row < rows; ++row) {
    uint32_t rest = row; bool inb = true; int64_t base = (int64_t)m.addr;
    for (int i = 1; i < r; ++i) {
      uint32_t j = rest % cnt[i]; rest /= cnt[i];
      int64_t xi = (int64_t)coords[i] + (int64_t)j * m.es[i];
      inb = inb && xi >= 0 && xi < (int64_t)m.dims[i];
      base += xi * (int64_t)m.strides[i - 1];
    }
    for (uint32_t e = 0; e < cnt[0]; ++e) {
      int64_t xe = (int64_t)coords[0] + e;
      bool in = inb && xe >= 0 && xe < (int64_t)m.dims[0];
      f(in ? (uint64_t)(base + xe * m.esz) : 0ull, tmaSwizzle(m.swizzle, smemOff + row * rowBytes + e * m.esz), in);
    }
  }
  return rows * rowBytes;
}

// `UTMALDG`: the box into shared memory at `smem` (zeros outside the tensor), then a complete-tx
// of its bytes on the mbarrier at `mbar`.
__device__ inline void tmaLoad(const Ctx& x, const TensorMapView& m, const int32_t* coords, uint32_t smem,
                               uint32_t mbar, uint32_t* fault) {
  if (!m.ok) return;
  uint8_t* win = sharedAt(x, smem) - (smem & 0xffffffu);
  int esz = m.esz;
  uint32_t bytes = tile(m, coords, smem & 0xffffffu, [&](uint64_t g, uint32_t off, bool in) {
    for (int b = 0; b < esz; ++b) win[off + b] = in ? gptr(x, g)[b] : 0;
  });
  __threadfence();
  mbAtomic(MB(x, mbar), 2, 0, (int64_t)bytes, false, fault);
}

// Bulk copies (`UBLKCP`, `NV.SASS.Async.ublkcp`): `size` bytes, complete when issued.
// `.S.G`: global to shared, then a complete-tx of the bytes on the mbarrier at `mbar`.
__device__ inline void bulkLoad(const Ctx& x, uint32_t smem, uint64_t g, uint32_t size, uint32_t mbar, uint32_t* fault) {
  uint8_t* d = sharedAt(x, smem);
  for (uint32_t i = 0; i < size; ++i) d[i] = gptr(x, g)[i];
  __threadfence();
  mbAtomic(MB(x, mbar), 2, 0, (int64_t)size, false, fault);
}
// `.G.S`: shared to global.
__device__ inline void bulkStore(const Ctx& x, uint64_t g, uint32_t smem, uint32_t size) {
  const uint8_t* s = sharedAt(x, smem);
  for (uint32_t i = 0; i < size; ++i) gptr(x, g)[i] = s[i];
  __threadfence();
}
// `.S.S`: this CTA's shared memory into a cluster CTA's at `dst`, its mbarrier at `mbar` signalled.
__device__ inline void bulkPeer(const Ctx& x, uint32_t dst, uint32_t src, uint32_t size, uint32_t mbar, uint32_t* fault) {
  uint8_t* d = sharedAt(x, dst);
  const uint8_t* s = sharedAt(x, src);
  for (uint32_t i = 0; i < size; ++i) d[i] = s[i];
  __threadfence();
  mbAtomic(MB(x, mbar), 2, 0, (int64_t)size, false, fault);
}

// One element of a bulk reduction (`UBLKRED`): `op` 0 add, 1 min, 2 max, 3 inc, 4 dec, 5 and,
// 6 or, 7 xor on type `T`, atomically (other CTAs may reduce into the same element).
template <class T>
__device__ inline void redElem(T* d, T v, int op) {
  switch (op) {
    case 0: atomicAdd(d, v); break;
    case 1: atomicMin(d, v); break;
    case 2: atomicMax(d, v); break;
    case 3: atomicInc((unsigned*)d, (unsigned)v); break;
    case 4: atomicDec((unsigned*)d, (unsigned)v); break;
    case 5: atomicAnd(d, v); break;
    case 6: atomicOr(d, v); break;
    default: atomicXor(d, v); break;
  }
}
template <>
__device__ inline void redElem<float>(float* d, float v, int op) { atomicAdd(d, v); }
template <>
__device__ inline void redElem<double>(double* d, double v, int op) { atomicAdd(d, v); }

// `UBLKRED.G.S` (global destination `g`) or `.S.S` (a cluster CTA's shared memory at `dst`, its
// mbarrier at `mbar` signalled): each element of the source at `src` reduced into the destination.
template <class T>
__device__ inline void bulkRed(const Ctx& x, uint8_t* d, uint32_t src, uint32_t size, int op) {
  const uint8_t* s = sharedAt(x, src);
  for (uint32_t i = 0; i < size; i += sizeof(T)) redElem<T>((T*)(d + i), *(const T*)(s + i), op);
  __threadfence();
}

// `UTMASTG`: the box's elements inside the tensor from shared memory at `smem`.
__device__ inline void tmaStore(const Ctx& x, const TensorMapView& m, const int32_t* coords, uint32_t smem,
                                uint32_t* fault) {
  if (!m.ok) return;
  const uint8_t* win = sharedAt(x, smem) - (smem & 0xffffffu);
  int esz = m.esz;
  tile(m, coords, smem & 0xffffffu, [&](uint64_t g, uint32_t off, bool in) {
    if (in) for (int b = 0; b < esz; ++b) gptr(x, g)[b] = win[off + b];
  });
  __threadfence();
}

}  // namespace lift

// ---------------------------------------------------------------------------------------------
// The lifted kernel's frame.

#define PROLOGUE                                                                                 \
  __shared__ uint32_t s_live[LIFT_WARPS];                                                        \
  const uint32_t tidx = threadIdx.x, tidy = threadIdx.y, tidz = threadIdx.z;                     \
  const uint32_t flat = tidx + blockDim.x * (tidy + blockDim.y * tidz);                          \
  const uint32_t lane = flat % 32, warp = flat / 32;                                             \
  const uint32_t cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);           \
  lift::Ctx ctx = ctx0;                                                                          \
  ctx.shared = ctx0.sharedAll + (uint64_t)cta * ctx0.sharedBytes;                                \
  {                                                                                              \
    uint32_t _bx = blockIdx.x - blockIdx.x % ctx0.cx, _by = blockIdx.y - blockIdx.y % ctx0.cy;     \
    uint32_t _bz = blockIdx.z - blockIdx.z % ctx0.cz;                                            \
    ctx.clusterRank = blockIdx.x % ctx0.cx + ctx0.cx * (blockIdx.y % ctx0.cy + ctx0.cy * (blockIdx.z % ctx0.cz)); \
    ctx.clusterBase = _bx + gridDim.x * (_by + gridDim.y * _bz);                                   \
  }                                                                                              \
  uint32_t cgaPhase = 0;                                                                         \
  ctx.tmem = ctx0.tmemAll + (uint64_t)cta * lift::TMEM_BYTES;                                    \
  uint32_t r[256], urf[64];                                                                      \
  uint32_t breg[16];                                                                             \
  uint8_t lmem[LIFT_LOCAL];                                                                      \
  bool p[8], upf[8];                                                                             \
  for (int _i = 0; _i < 256; ++_i) r[_i] = 0;                                                    \
  for (int _i = 0; _i < 64; ++_i) urf[_i] = 0;                                                   \
  for (int _i = 0; _i < 8; ++_i) { p[_i] = false; upf[_i] = false; }                             \
  if (lane == 0) {                                                                               \
    uint32_t _n = blockDim.x * blockDim.y * blockDim.z - warp * 32;                              \
    s_live[warp] = _n >= 32 ? 0xffffffffu : ((1u << _n) - 1u);                                   \
  }                                                                                              \
  __syncthreads();

#define EPILOGUE return;

// Uniform registers and predicates: each thread's own copy, written by every executing lane with
// the same value (a lane that does not execute a uniform write keeps its old value).
#define UR(i) (urf[(i)])
#define URW(i, e) do { uint32_t _v = (e); urf[(i)] = _v; } while (0)
#define UP(i) (upf[(i)])
#define UPW(i, e) do { bool _v = (e); upf[(i)] = _v; } while (0)
// B registers (`BSSY`'s lanes): each thread's own copy, so lanes that independent thread scheduling
// runs apart cannot overwrite each other's.
#define BREG(b) (breg[(b)])
#define LIVE (s_live[warp])
#define C32(off) lift::ld32c(ctx, (uint32_t)(off))
// A shared address names the CTA by its cluster rank in bits 24 and up (`Exec.sharedAt`): the
// executing CTA's own rank for its own memory.
#define SH(a) (lift::sharedAt(ctx, (uint32_t)(a)))
// A CTA's shared window is touched by the CTA alone, on one SM: its accesses are relaxed at CTA
// scope, which the SM's L1 serves (volatile accesses would go to the L2 every time).
#define LD_SH(T, PT, p) ([&]() { T _v; asm volatile("ld.relaxed.cta.global." PT " %0, [%1];" : "=r"(_v) : "l"(p) : "memory"); return _v; }())
#define ST_SH(PT, p, v) asm volatile("st.relaxed.cta.global." PT " [%0], %1;" :: "l"(p), "r"(v) : "memory")
// Local memory: the thread's own window, from address 0 (the simulator's per-thread store).
#define LOC(a) (lmem + (a))
// `Exec.genericTarget`: a generic address in the shared window (SWINHI over a shared address), the
// thread's local window (its base in constant bank 0 at LIFT_LOCALWIN, 16 MB), or global memory.
#define GEN(a) ([&](uint64_t _a) -> uint8_t* {                                                   \
    if ((uint32_t)(_a >> 32) == SWINHI) return lift::sharedAt(ctx, (uint32_t)_a);                \
    uint64_t _lw = lift::U64(C32(LIFT_LOCALWIN), C32(LIFT_LOCALWIN + 4));                         \
    if (_lw != 0 && _a >= _lw && _a < _lw + (1ull << 24)) return lmem + (_a - _lw);               \
    return lift::gptr(ctx, _a);                                                                  \
  }((uint64_t)(a)))
#define G(a) lift::gptr(ctx, (a))
#define FAULT (*ctx.fault)
// `Exec.sharedWindowHi`: the high word of generic shared addresses (`NV.Windows.shared`).
#define SWINHI 0x7ff1u
