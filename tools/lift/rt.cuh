// The accelerated mode's device runtime: what a lifted kernel (tools/lift/lift.py) calls, each a
// transcription of the simulator's definition so the results are the simulator's bit for bit.
//
// - The tensor cores' datapath on words (NV.Num.TensorModel's `Datapath`, NV/Num/Tensor.lean) and
//   warpgroup MMAs on it (NV.Async.Wgmma: descriptors, swizzle, the accumulator fragment; the
//   Hopper bf16/f16 → f32 model), in tiles of 16 output rows (the work pool, below).
// - TMA tiled copies (NV.Async.Tma: `TensorMap.tile`); tensor maps in the simulator's encoding
//   (NV.Async.TensorMap.encode), from the parameter block or global memory.
// - The work pool: one instruction's work (an MMA's output tiles, a copy's rows) spread over the
//   issuing warp and helper CTAs, which run on the SMs the launch leaves idle.
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
  // The work pool (`Pool`): the launch's CTAs, then its helper CTAs; this CTA's index.
  uint32_t ctas, helpers, cta;
  struct Pool* pool;
  struct Slot* slots;      // one per lifted warp (`cta * LIFT_WARPS + warp`), with helpers only
  uint32_t* acc;           // per slot, a warpgroup MMA's 16 x 256 accumulator words for its helpers
  unsigned long long* ring;  // published jobs, `ringSize` positions
  uint32_t ringSize;
  uint32_t* stage;         // this warp's (the issuer's) stage in shared memory (`STAGE_WORDS`)
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

// `UCGABAR_ARV`: release, then the warp's arrival (its lowest live lane's); returns the phase it
// arrived in, to every live lane. The live lanes `live` arrive together, once: `UCGABAR_ARV` is a
// warp-wide instruction, and lanes that reached it apart would each arrive.
__device__ inline uint32_t clusterArrive(const Ctx& x, uint32_t live) {
  __syncwarp(live);
  __threadfence();
  uint32_t ph = 0;
  if (threadIdx.x % 32 == __ffs(live) - 1) ph = clusterStep(clusterBar(x), true);
  return __shfl_sync(live, ph, __ffs(live) - 1);
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
#define MUFU_SQRT MufuPoly{kSqrt, 33, 18, false, true, 58, 0x7fff800000000000ull}

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

// `MUFU.SQRT` (Mufu.sqrt on a flushed input).
__device__ inline uint32_t mufuSqrt(uint32_t x) {
  x = flush32(x);
  bool neg = x >> 31;
  uint32_t ef = (x >> 23) & 0xff;
  if (isNaN32(x)) return 0x7fffffffu;
  if (ef == 0xff) return neg ? 0x7fffffffu : 0x7f800000u;
  if ((x & 0x7fffffffu) == 0) return x;
  if (neg) return 0x7fffffffu;
  int l = (int)ef - 127;
  uint32_t fr = x & 0x7fffff;
  int index = fr >> 17;
  uint64_t xl = fr & 0x1ffff;
  if (l & 1) { index += 64; l -= 1; }
  const MufuPoly P = MUFU_SQRT;
  uint64_t sum = mufuEval(P, index, xl, mufuSquare(xl));
  return mufuLdexp(0x3f800000u | mufuFrac(P, sum), l / 2);
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

// A codec whose element sits above `pad` ignored low bits (TF32 in a binary32 word: 13).
__device__ __forceinline__ Codec codecPad(int exp, int man, int pad) {
  Codec c = codecIeee(exp, man); c.pad = pad; return c;
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

// ---------------------------------------------------------------------------------------------
// 5th-generation tensor cores (`NV.Async.Tcgen05`, `NV.SASS.Async`).

// `Tcgen05.rowLane`: the lane of row `i` of an M-row operand from lane `l0`.
__device__ __forceinline__ uint32_t rowLane(uint32_t m, uint32_t l0, uint32_t i) {
  return m == 128 ? l0 + i : l0 + 32 * (i / 16) + i % 16;
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
static __device__ __noinline__ void tmemMove(const Ctx& x, bool ld, int shape, uint32_t regs, bool pack, uint32_t taddr,
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

// A box's shape (`TensorMap.tile`): its rows (every dimension past the first) of `rowBytes`.
struct BoxShape { uint32_t cnt[5]; uint32_t rowBytes, rows; };

__device__ inline BoxShape boxShape(const TensorMapView& m) {
  BoxShape b;
  for (int i = 0; i < 5; ++i) b.cnt[i] = i < m.rank ? (m.box[i] + m.es[i] - 1) / (m.es[i] ? m.es[i] : 1) : 1;
  b.rowBytes = b.cnt[0] * m.esz;
  b.rows = 1;
  for (int i = 1; i < m.rank; ++i) b.rows *= b.cnt[i];
  return b;
}

// Row `row` of the box at `coords`: its global address at element 0 of the box (`base`), and
// whether it lies inside the tensor in every dimension past the first.
__device__ inline bool boxRow(const TensorMapView& m, const BoxShape& b, const int32_t* coords, uint32_t row,
                              int64_t* base) {
  uint32_t rest = row; bool inb = true; int64_t a = (int64_t)m.addr;
  for (int i = 1; i < m.rank; ++i) {
    uint32_t j = rest % b.cnt[i]; rest /= b.cnt[i];
    int64_t xi = (int64_t)coords[i] + (int64_t)j * m.es[i];
    inb = inb && xi >= 0 && xi < (int64_t)m.dims[i];
    a += xi * (int64_t)m.strides[i - 1];
  }
  *base = a;
  return inb;
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


// ---------------------------------------------------------------------------------------------
// Warp-level matrix multiply-accumulate (`mma.sync`, NV.SASS.decodeMma) and matrix moves
// (`ldmatrix`/`stmatrix`, lifted inline by lift.py).

// The tensor-core datapath `mma.sync` into binary32 uses on SM version `sm`
// (`NV.PTX.Sem.Matrix.model`: `NV.Num.{Ampere,Hopper,Blackwell}.{f16f32,tf32f32}`): products per
// fused step, aligned fraction bits, the alignment's smallest exponent. Blocks of 0: none.
struct MmaModel { int blockSize, alignFrac, minAlignExp; };

__device__ __forceinline__ MmaModel mmaModel(uint32_t sm, bool tf32) {
  if (sm >= 80 && sm < 90) return tf32 ? MmaModel{4, 24, -132} : MmaModel{8, 24, -132};
  if (sm == 90 || (sm >= 100 && sm < 130)) return tf32 ? MmaModel{8, 25, -133} : MmaModel{16, 25, -133};
  return MmaModel{0, 0, 0};
}

// Whether this GPU's own `mma.sync` computes as datapath `m` does: its model (`mmaModel` of the
// architecture compiled for) the same. Then an emulated MMA runs on the tensor cores and gives the
// simulator's result (the models are the simulator's, which the hardware matches bit for bit);
// else in software. `LIFT_SOFT_MMA` keeps every MMA in software.
__device__ __forceinline__ bool nativeMma(MmaModel m, bool tf32) {
#if defined(__CUDA_ARCH__) && !defined(LIFT_SOFT_MMA)
  const MmaModel h = mmaModel(__CUDA_ARCH__ / 10, tf32);
  return h.blockSize != 0 && h.blockSize == m.blockSize && h.alignFrac == m.alignFrac && h.minAlignExp == m.minAlignExp;
#else
  return false;
#endif
}

// `mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}` on this GPU, `d` the accumulator in and out.
__device__ __forceinline__ void mma16816(bool bf, const uint32_t (&a)[4], const uint32_t (&b)[2], uint32_t (&d)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if (bf)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// The other `mma.sync` shapes into binary32 (`hmma`'s): m16n8k8 of f16 or bf16 (`w` 16) and
// m16n8k8 and m16n8k4 of tf32 (`w` 32), on this GPU.
template <int K, int AW, int BW>
__device__ __forceinline__ void mmaOther(int w, bool bf, const uint32_t (&a)[AW], const uint32_t (&b)[BW], uint32_t (&d)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (AW == 2 && BW == 1) {
    if (w == 32)
      asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                   : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    else if (bf)
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                   : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    else
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                   : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(b[0]));
  } else if constexpr (AW == 4 && BW == 2) {
    if (w == 32)
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                   : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    else
      mma16816(bf, a, b, d);
  }
#endif
}

// `HMMA.{16816|1688|1684}.F32[.BF16|.TF32]` by the whole warp: D = A·B + C, 16 x 8 outputs of K
// products each, every output `Datapath.dot` over blocks of the model's size from C's value
// (`NV.Num.TensorModel.tileWords`). Lane (g, q) = (lane / 4, lane % 4) holds the PTX fragments
// (`aCoord`, `bCoord`, `cCoord`): A's element i in row g + 8·(r mod 2), column q·per + i mod per +
// 4·per·⌊r/2⌋ (r = ⌊i/per⌋, per elements of `w` bits a register); B's in row q·per + i mod per +
// 4·per·r, column g; C's and D's value i in row g + 8·⌊i/2⌋, column 2q + i mod 2. The operands are
// encoded once (`Codec.encode`, `c`; `man` fraction bits) into the warp's stage, A by row and B by
// column.
template <int K, int AW, int BW>
static __device__ __noinline__ void hmma(const Ctx& x, MmaModel m, int w, const Codec& c, int man, const uint32_t (&a)[AW],
                            const uint32_t (&b)[BW], const uint32_t (&cin)[4], uint32_t (&d)[4], uint32_t lane,
                            uint32_t mask) {
  if (mask != 0xffffffffu || m.blockSize == 0) {
    if (lane == __ffs(mask) - 1) atomicOr(x.fault, 32u);
    return;
  }
  if (nativeMma(m, w == 32)) {
    for (int i = 0; i < 4; ++i) d[i] = cin[i];
    mmaOther<K, AW, BW>(w, man == 7, a, b, d);
    return;
  }
  uint32_t* A = x.stage;
  uint32_t* B = x.stage + 16 * K;
  const int per = 32 / w;
  const uint32_t g = lane / 4, q = lane % 4;
  __syncwarp();
  for (int i = 0; i < AW * per; ++i) {
    const uint32_t e = w == 32 ? a[i] : (a[i / per] >> (w * (i % per))) & ((1u << w) - 1);
    const int r = i / per;
    A[(g + 8 * (r % 2)) * K + q * per + i % per + 4 * per * (r / 2)] = encode(c, e);
  }
  for (int i = 0; i < BW * per; ++i) {
    const uint32_t e = w == 32 ? b[i] : (b[i / per] >> (w * (i % per))) & ((1u << w) - 1);
    const int r = i / per;
    B[g * K + q * per + i % per + 4 * per * r] = encode(c, e);
  }
  __syncwarp();
  const Datapath dp = prepareF32(m.alignFrac, m.minAlignExp, 23, false, man, man);
  for (int i = 0; i < 4; ++i) {
    const uint32_t row = g + 8 * (i / 2), col = 2 * q + i % 2;
    uint64_t acc = cin[i];
    for (int s = 0; s < K; s += m.blockSize) acc = blockAt(dp, A + row * K + s, B + col * K + s, min(m.blockSize, K - s), acc);
    d[i] = (uint32_t)acc;
  }
  __syncwarp();
}

// `BSYNC` in a kernel with `BREAK` (`NV.SASS.Sem`: a `BREAK` takes its lanes out of the barrier's
// mask): this lane arrives at barrier `bar` (arrived lanes, broken-out lanes) of mask `m`, waits
// until every lane of the mask that has not broken out or exited (`live`) has arrived, and the
// lanes leave together.
__device__ inline uint32_t bsync(uint32_t* bar, uint32_t m, const volatile uint32_t* live, uint32_t lane) {
  atomicOr(&bar[0], 1u << lane);
  uint32_t need;
  for (unsigned ns = 8;; ns = min(2 * ns, 256u)) {
    need = m & *live & ~*(volatile uint32_t*)&bar[1];
    if ((*(volatile uint32_t*)&bar[0] & need) == need) break;
    __nanosleep(ns);
  }
  __syncwarp(need);
  return need;
}

// `.SAT` on a binary32 result (`NV.Num.satUnit`): NaN and negatives (-0 too) to +0, above one to one.
__device__ __forceinline__ uint32_t sat32(uint32_t u) {
  if ((u & 0x7fffffffu) > 0x7f800000u || (u >> 31)) return 0u;
  return u > 0x3f800000u ? 0x3f800000u : u;
}

// The heavy helpers (a warpgroup or tensor-memory MMA, a TMA copy, the work pool's tasks) are kept
// out of line and static: compiled once a kernel, not at every call site (a TMA and wgmma GEMM's
// lifted kernel compiled in 18 s inlined, 3 s out of line), and not at all when unused.
// ---------------------------------------------------------------------------------------------
// The work pool. An instruction whose work the simulator does in one step (a warpgroup or
// tensor-memory MMA, a TMA or bulk copy) is a job of independent tasks: an MMA's tiles of 16 output
// rows, a copy's 4 KB of rows. With no helpers the issuer runs every task itself. With helpers (CTAs
// appended to the grid on the SMs the launch leaves idle; main.cu launches the grid cooperatively,
// so they are all resident) the issuer publishes the job on the ring, works its tasks beside them
// when it is a whole warp, and waits for all of them, so the instruction still completes when it
// issues. Every output word or byte is one task's, computed from the same inputs by the same
// functions wherever it runs, so the results do not depend on who ran which task.

constexpr uint32_t NONE = 0xffffffffu;
// Per warp, in dynamic shared memory: an MMA task's operands decoded once (A and B, 16 rows or
// columns of 16), and a helper's copy of the job it works on.
constexpr uint32_t STAGE_WORDS = 512, JOB_WORDS = 192;
constexpr uint32_t WARP_SMEM = 4 * (STAGE_WORDS + JOB_WORDS);
constexpr uint32_t COPY_TASK_BYTES = 4096;
// A warpgroup MMA's accumulator words per warp, for its helpers (16 rows of N <= 256).
constexpr uint32_t ACC_WORDS = 16 * 256;

// An MMA: `rows x cols` outputs, each a 16-term dot product (`Datapath.blockAt`) onto its
// accumulator, in tiles of 16 rows by `tileCols`.
struct MmaJob {
  uint64_t da, db;
  uint32_t aRow0;           // the first A row (a warpgroup MMA: the warp's 16 of the 64)
  uint32_t aSplit, bSplit;  // A rows, B columns from these on are the next cluster rank's (`.2CTA`); NONE
  uint32_t rows, cols, tileCols;
  uint32_t dl, dc, M;       // into tensor memory (`toTmem`): its lane and column, the MMA's M
  uint8_t ty, aK, bK, negA, negB, acc, toTmem, pair;
};

// A copy: a TMA box (`kind` 0 load, 1 store) or bytes (`kind` 2).
struct CopyJob {
  TensorMapView m;
  int32_t c[5];
  uint32_t kind;
  uint32_t smem;            // the box's shared address
  uint32_t ranks;           // a load's destinations: with bit 16 (`.MULTICAST`), bit k names cluster
                            // rank k, at `smem`'s offset; else `smem` itself
  uint8_t* dst;             // kind 2
  const uint8_t* src;
  uint32_t size;
};

struct Job {
  Ctx x;                    // the issuer's
  uint32_t kind, tasks;     // kind 0 MMA, 1 copy
  uint32_t* buf;            // an MMA's outputs outside tensor memory: `cols` words a row
  union { MmaJob mma; CopyJob copy; };
};
static_assert(sizeof(Job) <= 4 * JOB_WORDS && sizeof(Job) % 8 == 0, "a job fits a warp's copy");

// One per lifted warp: its job; the job's claims, `(generation << 32) | next task`; the tasks
// others finished.
struct Slot {
  unsigned long long claim;
  unsigned done, pad;
  Job job;
};

// The launch's pool: the ring's head (the first position no helper has passed) and tail, each
// entry `position << 20 | slot`; the lifted warps not yet exited. Each on its own line.
struct Pool {
  unsigned long long head, pad0[15];
  unsigned long long tail, pad1[15];
  unsigned live, pad2[31];
};

__device__ __forceinline__ unsigned long long ldAcq(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ldAcq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void stRel(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// A lifted warp's exit: out of its cluster's live warps and the pool's.
__device__ inline void warpExit(const Ctx& x) {
  clusterExit(x);
  if (x.helpers) atomicSub(&x.pool->live, 1u);
}

// Element (mn, k) of an MMA's A or B (`Desc.addr`), encoded for the datapath (`Codec.encode`):
// from the issuer's CTA's shared memory or, past the split, the next cluster rank's.
__device__ inline uint32_t mmaRaw(const Ctx& x, const MmaJob& j, bool isA, uint32_t mn, int k) {
  uint32_t split = isA ? j.aSplit : j.bSplit, rank = x.clusterRank;
  if (mn >= split) { mn -= split; rank += 1; }
  uint32_t a = descAddr(isA ? j.da : j.db, isA ? j.aK : j.bK, (int)mn, k, x.fault);
  uint16_t v = __ldcg((const unsigned short*)sharedAt(x, (rank << 24) | a));
  if (isA ? j.negA : j.negB) v ^= 0x8000;
  return v;
}
__device__ inline uint32_t mmaElem(const Ctx& x, const MmaJob& j, bool isA, uint32_t mn, int k, const Codec& c) {
  return encode(c, mmaRaw(x, j, isA, mn, k));
}

// Output `o` of a tile, as row `i` and column `k`: in each 16 x 8 block the warpgroup MMA's
// accumulator fragment (`accPos`), so a warp running a block computes each lane's own values (lane
// `o % 32`, its value `4 · block + o / 32 % 4`); a 16 x 16 tile is two blocks.
__device__ __forceinline__ void tilePos(uint32_t o, uint32_t* i, uint32_t* k) {
  uint32_t l = o % 32, q = (o / 32) % 4;
  *i = l / 4 + 8 * (q / 2);
  *k = 2 * (l % 4) + q % 2 + 8 * (o / 128);
}

// The word output (row, col) of an MMA lands in: `buf`'s, or tensor memory's (`UTCHMMA`'s rows:
// a pair's rows from 128 on in the peer's).
__device__ inline uint32_t* mmaOut(const Ctx& x, const MmaJob& j, uint32_t* buf, uint32_t row, uint32_t col) {
  if (!j.toTmem) return buf + row * j.cols + col;
  uint32_t b = j.pair && row >= 128 ? ((x.clusterRank + 2) << 24) + tmemByte(row - 128, j.dc + col)
                                    : tmemByte(j.pair ? row : rowLane(j.M, j.dl, row), j.dc + col);
  return (uint32_t*)tmemAt(x, b);
}

// MMA task `t`, a tile, run by `n` lanes of one warp (`r` this lane's rank, `sync` the warp's
// barrier) with its operands decoded once into the warp's stage `st`. `own`: the lanes'
// accumulators, when the issuing warp runs a warpgroup MMA's task whole (its outputs are the lanes'
// own values, `tilePos`).
template <class Sync>
static __device__ __noinline__ void mmaTask(const Ctx& x, const MmaJob& j, uint32_t* buf, uint32_t t, uint32_t r, uint32_t n,
                               uint32_t* st, Sync sync, uint32_t* own) {
  const int man = j.ty == BF16 ? 7 : 10;
  const Codec c = j.ty == BF16 ? codecIeee(8, 7) : codecIeee(5, 10);
  const Datapath dp = prepareF32(25, -133, 23, false, man, man);
  const uint32_t blocks = (j.cols + j.tileCols - 1) / j.tileCols;
  const uint32_t row0 = 16 * (t / blocks), col0 = j.tileCols * (t % blocks);
  const uint32_t nr = min(16u, j.rows - row0), nc = min(j.tileCols, j.cols - col0);
  // A whole warp on a GPU whose `mma.sync` computes as this datapath does: each 16 x 8 block of
  // the tile one `mma.sync.m16n8k16`, its fragments the lanes' (`tilePos`: block-major, as
  // `accPos`), rows and columns past the tile's edge zero and not stored.
  if (n == 32 && nativeMma(MmaModel{16, 25, -133}, false)) {
    const uint32_t g = r / 4, q = r % 4;
    uint32_t a[4];
    for (int h = 0; h < 4; ++h) {
      const uint32_t i = g + 8 * (h % 2);
      const int k = 2 * q + 8 * (h / 2);
      a[h] = i < nr ? mmaRaw(x, j, true, j.aRow0 + row0 + i, k) | mmaRaw(x, j, true, j.aRow0 + row0 + i, k + 1) << 16 : 0u;
    }
    for (uint32_t c0 = 0; c0 < j.tileCols; c0 += 8) {
      uint32_t b[2], d[4];
      const uint32_t col = c0 + g;
      for (int h = 0; h < 2; ++h) {
        const int k = 2 * q + 8 * h;
        b[h] = col < nc ? mmaRaw(x, j, false, col0 + col, k) | mmaRaw(x, j, false, col0 + col, k + 1) << 16 : 0u;
      }
      for (int v = 0; v < 4; ++v) {
        const uint32_t i = g + 8 * (v / 2), k = c0 + 2 * q + v % 2;
        d[v] = 0u;
        if (j.acc && i < nr && k < nc)
          d[v] = own ? own[4 * ((col0 + k) / 8) + v] : __ldcg(mmaOut(x, j, buf, row0 + i, col0 + k));
      }
      mma16816(j.ty == BF16, a, b, d);
      for (int v = 0; v < 4; ++v) {
        const uint32_t i = g + 8 * (v / 2), k = c0 + 2 * q + v % 2;
        if (i >= nr || k >= nc) continue;
        if (own) own[4 * ((col0 + k) / 8) + v] = d[v];
        else *mmaOut(x, j, buf, row0 + i, col0 + k) = d[v];
      }
    }
    return;
  }
  uint32_t* A = st;
  uint32_t* B = st + 256;
  sync();
  for (uint32_t e = r; e < 16 * (nr + nc); e += n) {
    if (e < 16 * nr) A[e] = mmaElem(x, j, true, j.aRow0 + row0 + e / 16, (int)(e % 16), c);
    else B[e - 16 * nr] = mmaElem(x, j, false, col0 + (e - 16 * nr) / 16, (int)((e - 16 * nr) % 16), c);
  }
  sync();
  for (uint32_t o = r; o < 16 * j.tileCols; o += n) {
    uint32_t i, k;
    tilePos(o, &i, &k);
    if (i >= nr || k >= nc) continue;
    if (own) {
      uint32_t v = 4 * ((col0 + k) / 8) + (o / 32) % 4;
      own[v] = (uint32_t)blockAt(dp, A + 16 * i, B + 16 * k, 16, j.acc ? own[v] : 0u);
    } else {
      uint32_t* d = mmaOut(x, j, buf, row0 + i, col0 + k);
      *d = (uint32_t)blockAt(dp, A + 16 * i, B + 16 * k, 16, j.acc ? __ldcg(d) : 0u);
    }
  }
}

// Up to 16 bytes from `s` to `d`: one move when whole and both are aligned, else byte by byte.
// Loads from the L2 (another SM may have written them, the issuer's or a helper's); plain stores,
// which keep this SM's L1 coherent with them.
__device__ __forceinline__ void copy16(uint8_t* d, const uint8_t* s, uint32_t n) {
  if (n == 16 && ((uintptr_t)d & 15) == 0 && ((uintptr_t)s & 15) == 0) *(uint4*)d = __ldcg((const uint4*)s);
  else for (uint32_t b = 0; b < n; ++b) d[b] = __ldcg(s + b);
}

// A box copy's rows per task: about 4 KB of them, at least one.
__device__ __forceinline__ uint32_t rowsPerTask(const BoxShape& b) { return max(1u, COPY_TASK_BYTES / max(b.rowBytes, 1u)); }

// Copy task `t` by `n` lanes (`r` this lane's rank): 4 KB of bytes, or the box's rows in as many
// bytes, a lane a 16-byte unit of a row at a time (`TensorMap.tile`: elements outside the tensor
// read as zeros and are not stored; the swizzle permutes whole 16-byte units).
static __device__ __noinline__ void copyTask(const Ctx& x, const CopyJob& j, uint32_t t, uint32_t r, uint32_t n) {
  if (j.kind == 2) {
    const uint32_t lo = t * COPY_TASK_BYTES, hi = min(j.size, lo + COPY_TASK_BYTES);
    for (uint32_t b = lo + 16 * r; b < hi; b += 16 * n) copy16(j.dst + b, j.src + b, min(16u, hi - b));
    return;
  }
  const TensorMapView& m = j.m;
  const BoxShape bs = boxShape(m);
  const uint32_t per = rowsPerTask(bs), units = (bs.rowBytes + 15) / 16;
  const uint32_t row0 = t * per, rows = min(per, bs.rows - row0), off0 = j.smem & 0xffffffu;
  uint8_t* own = sharedAt(x, j.smem) - off0;  // the window holding the box
  // A load's destination windows.
  auto each = [&](auto f) {
    if (j.kind != 0 || !(j.ranks & 0x10000u)) { f(own); return; }
    for (uint32_t k = 0; k < 16; ++k)
      if ((j.ranks >> k) & 1u) f(sharedAt(x, (k << 24) | off0) - off0);
  };
  for (uint32_t u = r; u < rows * units; u += n) {
    const uint32_t row = row0 + u / units, b0 = 16 * (u % units), nb = min(16u, bs.rowBytes - b0);
    int64_t base;
    const bool inb = boxRow(m, bs, j.c, row, &base);
    const uint32_t lin = off0 + row * bs.rowBytes + b0;
    const int64_t x0 = (int64_t)j.c[0] + b0 / m.esz, x1 = x0 + nb / m.esz;
    if (inb && x0 >= 0 && x1 <= (int64_t)m.dims[0] && (lin & 15) == 0) {
      uint8_t* g = gptr(x, (uint64_t)(base + x0 * m.esz));
      const uint32_t off = tmaSwizzle(m.swizzle, lin);
      if (j.kind == 0) each([&](uint8_t* w) { copy16(w + off, g, nb); });
      else copy16(g, own + off, nb);
      continue;
    }
    for (uint32_t e = b0 / m.esz; e < (b0 + nb) / m.esz; ++e) {
      const int64_t xe = (int64_t)j.c[0] + e;
      const bool in = inb && xe >= 0 && xe < (int64_t)m.dims[0];
      const uint32_t off = tmaSwizzle(m.swizzle, off0 + row * bs.rowBytes + e * m.esz);
      uint8_t* g = in ? gptr(x, (uint64_t)(base + xe * m.esz)) : nullptr;
      for (int b = 0; b < m.esz; ++b) {
        if (j.kind == 0) {
          const uint8_t v = in ? __ldcg(g + b) : 0;
          each([&](uint8_t* w) { w[off + b] = v; });
        } else if (in) {
          g[b] = __ldcg(own + off + b);
        }
      }
    }
  }
}

template <class Sync>
__device__ inline void runTask(const Job& j, uint32_t t, uint32_t r, uint32_t n, uint32_t* st, Sync sync,
                               uint32_t* own) {
  if (j.kind == 0) mmaTask(j.x, j.mma, j.buf, t, r, n, st, sync, own);
  else copyTask(j.x, j.copy, t, r, n);
}

// The next task of slot `s`'s job of generation `gen`, if one is left.
__device__ inline int claimTask(Slot* s, unsigned long long gen, uint32_t tasks) {
  unsigned long long c = ldAcq(&s->claim);
  while ((c >> 32) == gen && (uint32_t)c < tasks) {
    unsigned long long old = atomicCAS(&s->claim, c, c + 1);
    if (old == c) return (int)(uint32_t)c;
    c = old;
  }
  return -1;
}

// The issuer runs job `j`, which each of its `n` lanes (`mask`; `r`, this lane's rank) built alike
// in warp `warp`: with no helpers, every task itself; else published on the ring, its tasks claimed
// beside the helpers when `work`, and waited for. `back(mine)` then runs on every lane, `mine`
// the tasks below 64 the issuer ran itself.
template <class Sync, class Back>
static __device__ __noinline__ void runJob(const Ctx& x, const Job& j, uint32_t warp, uint32_t r, uint32_t n, uint32_t mask,
                              Sync sync, uint32_t* own, bool work, Back back) {
  if (x.helpers == 0) {
    for (uint32_t t = 0; t < j.tasks; ++t) runTask(j, t, r, n, x.stage, sync, own);
    return;
  }
  const uint32_t slot = x.cta * LIFT_WARPS + warp, leader = __ffs(mask) - 1;
  Slot* s = x.slots + slot;
  unsigned long long gen = 0;
  if (r == 0) {
    s->job = j;
    s->done = 0;
    gen = (ldAcq(&s->claim) >> 32) + 1;
    stRel(&s->claim, gen << 32);
    const unsigned long long pos = atomicAdd(&x.pool->tail, 1ull);
    for (unsigned ns = 32; pos - ldAcq(&x.pool->head) >= x.ringSize; ns = min(2 * ns, 1024u)) __nanosleep(ns);
    stRel(&x.ring[pos % x.ringSize], (pos << 20) | slot);
  }
  gen = __shfl_sync(mask, gen, leader);
  uint32_t self = 0;
  unsigned long long mine = 0;
  while (work) {
    int t = 0;
    if (r == 0) t = claimTask(s, gen, j.tasks);
    t = __shfl_sync(mask, t, leader);
    if (t < 0) break;
    runTask(j, (uint32_t)t, r, n, x.stage, sync, own);
    self += 1;
    if (t < 64) mine |= 1ull << t;
  }
  if (r == 0)
    for (unsigned ns = 32; ldAcq(&s->done) + self < j.tasks; ns = min(2 * ns, 256u)) __nanosleep(ns);
  sync();
  back(mine);
}

// A helper CTA's warps: the job at the ring's head, its tasks claimed one by one until none is
// left, then passed; asleep while the ring is empty; gone once every lifted warp has exited.
static __device__ __noinline__ void serve(const Ctx& x, uint32_t* smem) {
  const uint32_t threads = blockDim.x * blockDim.y * blockDim.z;
  const uint32_t flat = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const uint32_t lane = flat % 32, warp = flat / 32, n = min(32u, threads - 32 * warp);
  const uint32_t mask = n == 32 ? 0xffffffffu : (1u << n) - 1u;
  uint32_t* st = smem + warp * (STAGE_WORDS + JOB_WORDS);
  Job& jc = *(Job*)(st + STAGE_WORDS);
  auto sync = [&]() { __syncwarp(mask); };
  unsigned ns = 64;
  while (true) {
    unsigned long long h = 0, e = 0;
    unsigned live = 1;
    if (lane == 0) {
      h = ldAcq(&x.pool->head);
      e = ldAcq(&x.ring[h % x.ringSize]);
      if ((e >> 20) != h) live = ldAcq(&x.pool->live);
    }
    h = __shfl_sync(mask, h, 0);
    e = __shfl_sync(mask, e, 0);
    live = __shfl_sync(mask, live, 0);
    if ((e >> 20) != h) {
      if (live == 0) return;
      __nanosleep(ns);
      ns = min(2 * ns, 4096u);
      continue;
    }
    ns = 64;
    Slot* s = x.slots + (uint32_t)(e & 0xfffffu);
    unsigned long long c = 0;
    if (lane == 0) c = ldAcq(&s->claim);
    c = __shfl_sync(mask, c, 0);
    sync();
    for (uint32_t w = lane; w < sizeof(Job) / 8; w += n)
      ((unsigned long long*)&jc)[w] = __ldcg((const unsigned long long*)&s->job + w);
    sync();
    while (true) {
      int t = 0;
      if (lane == 0) t = claimTask(s, c >> 32, jc.tasks);
      t = __shfl_sync(mask, t, 0);
      if (t < 0) break;
      runTask(jc, (uint32_t)t, lane, n, st, sync, (uint32_t*)nullptr);
      __threadfence();
      sync();
      if (lane == 0) atomicAdd(&s->done, 1u);
    }
    if (lane == 0) atomicCAS(&x.pool->head, h, h + 1);
  }
}

// `HGMMA.64xNx16.F32[.BF16]` from shared memory, by the whole warp (`wgmma.mma_async.sync
// .aligned`), each of the warpgroup's warps its 16 rows: one job of N/8 tiles of 16 x 8, whose
// outputs are the lanes' own values (`tilePos`). Thread `t` of the warpgroup holds row
// `16·(t/32) + (t%32)/4 + 8·((v/2) mod 2)`, column `2·(t%4) + v mod 2 + 8·(v/4)` of its value v
// (`accPos`), in `acc`, updated in place (`scaleD` false: from zero). Hopper's datapath for
// f16/bf16 inputs and an f32 accumulator: blocks of 16, 25 fraction bits aligned, exponent floor
// -133, the sum truncated to binary32 (`Hopper.f16f32`).
template <int N>
static __device__ __noinline__ void hgmma(const Ctx& x, InType ty, uint64_t da, uint64_t db, bool aK, bool bK, bool negA,
                             bool negB, bool scaleD, uint32_t* acc, uint32_t lane, uint32_t warp, uint32_t mask) {
  if (mask != 0xffffffffu) { if (lane == __ffs(mask) - 1) atomicOr(x.fault, 32u); return; }
  // On a GPU whose `mma.sync` computes as Hopper's warpgroup MMA does: this warp's 16 rows as N/8
  // `mma.sync.m16n8k16`, whose accumulator fragments are the lanes' values here (`accPos`), the
  // operands read from the descriptors' elements.
  if (nativeMma(mmaModel(90, false), false)) {
    const uint32_t g = lane / 4, q = lane % 4, row0 = 16 * (warp % 4);
    auto el = [&](uint64_t desc, bool kMajor, bool neg, uint32_t mn, int k) {
      const uint32_t at = descAddr(desc, kMajor, (int)mn, k, x.fault);
      return (uint32_t)(__ldcg((const unsigned short*)sharedAt(x, (x.clusterRank << 24) | at)) ^ (neg ? 0x8000u : 0u));
    };
    uint32_t a[4];
    for (int r = 0; r < 4; ++r) {
      const uint32_t row = row0 + g + 8 * (r % 2);
      const int col = 2 * q + 8 * (r / 2);
      a[r] = el(da, aK, negA, row, col) | el(da, aK, negA, row, col + 1) << 16;
    }
    for (uint32_t nb = 0; nb < N / 8; ++nb) {
      uint32_t b[2], d[4];
      for (int r = 0; r < 2; ++r) {
        const int k = 2 * q + 8 * r;
        b[r] = el(db, bK, negB, 8 * nb + g, k) | el(db, bK, negB, 8 * nb + g, k + 1) << 16;
      }
      for (int v = 0; v < 4; ++v) d[v] = scaleD ? acc[4 * nb + v] : 0u;
      mma16816(ty == BF16, a, b, d);
      for (int v = 0; v < 4; ++v) acc[4 * nb + v] = d[v];
    }
    return;
  }
  Job j;
  j.x = x; j.kind = 0; j.tasks = N / 8;
  j.buf = x.helpers ? x.acc + (x.cta * LIFT_WARPS + warp) * ACC_WORDS : nullptr;
  MmaJob& m = j.mma;
  m.da = da; m.db = db; m.aRow0 = 16 * (warp % 4); m.aSplit = NONE; m.bSplit = NONE;
  m.rows = 16; m.cols = N; m.tileCols = 8; m.dl = 0; m.dc = 0; m.M = 64;
  m.ty = (uint8_t)ty; m.aK = aK; m.bK = bK; m.negA = negA; m.negB = negB; m.acc = scaleD; m.toTmem = 0; m.pair = 0;
  // Value v of this lane is output `lane + 32·(v mod 4)` of block `v/4`.
  auto at = [&](uint32_t v) {
    uint32_t i, k;
    tilePos(lane + 32 * (v % 4), &i, &k);
    return j.buf + i * N + 8 * (v / 4) + k;
  };
  // The helpers' tiles start from the accumulators in `buf`.
  if (x.helpers && scaleD)
    for (uint32_t v = 0; v < N / 2; ++v) *at(v) = acc[v];
  auto sync = [&]() { __syncwarp(mask); };
  runJob(x, j, warp, lane, 32, mask, sync, acc, true, [&](unsigned long long mine) {
    for (uint32_t v = 0; v < N / 2; ++v)
      if (!((mine >> (v / 4)) & 1ull)) acc[v] = __ldcg(at(v));
  });
}

// `UTCHMMA[.2CTA]`: one `tcgen05.mma` of `.kind::f16` (f16 or bf16 inputs into f32), A and B from
// shared memory, issued by one thread (`Tcgen05.Inst.op`): D = A·B (+ D), the B200's datapath
// (`NV.Num.Blackwell.f16f32`: Hopper's), in tiles of 16 x 16. `pair`: `.cta_group::2`, M 256, rows
// and columns 128/N2 on the peer (the next rank). Others set the fault flag.
static __device__ __noinline__ void utcmma(const Ctx& x, uint64_t da, uint64_t db, uint32_t d, uint32_t idesc, bool accumulate,
                              bool pair, uint32_t lane, uint32_t warp) {
  auto f = [&](int lo, int w) { return (idesc >> lo) & ((1u << w) - 1); };
  uint32_t dtype = f(4, 2), atype = f(7, 3), btype = f(10, 3);
  bool negA = f(13, 1), negB = f(14, 1), transA = f(15, 1), transB = f(16, 1);
  uint32_t N = f(17, 6) << 3, M = f(24, 5) << 4;
  bool bad = f(2, 1) || f(30, 2) || dtype != 1 || atype > 1 || btype != atype ||
             (pair ? M != 256 : (M != 64 && M != 128)) || N < 8 || N > 256 || ((da >> 61) & 1) || ((db >> 61) & 1) ||
             ((da >> 46) & 7) != 1 || ((db >> 46) & 7) != 1;
  uint32_t dl = d >> 16, dc = d & 0xffffu;
  if (bad || dc + N > 512 || (M == 64 ? (dl != 0 && dl != 16) : dl != 0)) { atomicOr(x.fault, 32u); return; }
  Job j;
  j.x = x; j.kind = 0; j.buf = nullptr; j.tasks = (M / 16) * ((N + 15) / 16);
  MmaJob& m = j.mma;
  m.da = da; m.db = db; m.aRow0 = 0; m.aSplit = pair ? 128u : NONE; m.bSplit = pair ? N / 2 : NONE;
  m.rows = M; m.cols = N; m.tileCols = 16; m.dl = dl; m.dc = dc; m.M = M;
  m.ty = (uint8_t)(atype == 1 ? BF16 : F16); m.aK = !transA; m.bK = !transB; m.negA = negA; m.negB = negB;
  m.acc = accumulate; m.toTmem = 1; m.pair = pair;
  runJob(x, j, warp, 0, 1, 1u << lane, [] {}, (uint32_t*)nullptr, false, [](unsigned long long) {});
  __threadfence();
}

// `UTMALDG`: the box into shared memory at `smem` (zeros outside the tensor), of this CTA or, with
// bit 16 of `ranks` (`.MULTICAST`), of each cluster rank its bits 0-15 name, at `smem`'s offset; then
// a complete-tx of its bytes on the mbarrier at `mbar` (at its offset in each). Issued by one lane.
static __device__ __noinline__ void tmaLoad(const Ctx& x, const TensorMapView& m, const int32_t* coords, uint32_t smem,
                               uint32_t mbar, uint32_t ranks, uint32_t lane, uint32_t warp) {
  if (!m.ok) return;
  const BoxShape b = boxShape(m);
  Job j;
  j.x = x; j.kind = 1; j.buf = nullptr;
  j.tasks = (b.rows + rowsPerTask(b) - 1) / rowsPerTask(b);
  CopyJob& c = j.copy;
  c.m = m; c.kind = 0; c.smem = smem; c.ranks = ranks;
  for (int i = 0; i < 5; ++i) c.c[i] = coords[i];
  runJob(x, j, warp, 0, 1, 1u << lane, [] {}, (uint32_t*)nullptr, true, [](unsigned long long) {});
  __threadfence();
  const int64_t bytes = (int64_t)b.rows * b.rowBytes;
  if (!(ranks & 0x10000u)) mbAtomic(MB(x, mbar), 2, 0, bytes, false, x.fault);
  else
    for (uint32_t k = 0; k < 16; ++k)
      if ((ranks >> k) & 1u) mbAtomic(MB(x, (k << 24) | (mbar & 0xffffffu)), 2, 0, bytes, false, x.fault);
}

// `UTMASTG`: the box's elements inside the tensor from shared memory at `smem`. Issued by one lane.
static __device__ __noinline__ void tmaStore(const Ctx& x, const TensorMapView& m, const int32_t* coords, uint32_t smem,
                                uint32_t lane, uint32_t warp) {
  if (!m.ok) return;
  const BoxShape b = boxShape(m);
  Job j;
  j.x = x; j.kind = 1; j.buf = nullptr;
  j.tasks = (b.rows + rowsPerTask(b) - 1) / rowsPerTask(b);
  CopyJob& c = j.copy;
  c.m = m; c.kind = 1; c.smem = smem; c.ranks = 0;
  for (int i = 0; i < 5; ++i) c.c[i] = coords[i];
  runJob(x, j, warp, 0, 1, 1u << lane, [] {}, (uint32_t*)nullptr, true, [](unsigned long long) {});
  __threadfence();
}

// Bulk copies (`UBLKCP`, `NV.SASS.Async.ublkcp`): `size` bytes from `src` to `dst` (shared to shared
// across the cluster, global to shared, or shared to global), complete when issued; then, unless
// `mbar` is NONE, a complete-tx of the bytes on the mbarrier at `mbar`. Issued by one lane.
static __device__ __noinline__ void bulkCopy(const Ctx& x, uint8_t* dst, const uint8_t* src, uint32_t size, uint32_t mbar,
                                uint32_t lane, uint32_t warp) {
  Job j;
  j.x = x; j.kind = 1; j.buf = nullptr; j.tasks = (size + COPY_TASK_BYTES - 1) / COPY_TASK_BYTES;
  CopyJob& c = j.copy;
  c.kind = 2; c.dst = dst; c.src = src; c.size = size;
  runJob(x, j, warp, 0, 1, 1u << lane, [] {}, (uint32_t*)nullptr, true, [](unsigned long long) {});
  __threadfence();
  if (mbar != NONE) mbAtomic(MB(x, mbar), 2, 0, (int64_t)size, false, x.fault);
}

}  // namespace lift

// ---------------------------------------------------------------------------------------------
// The lifted kernel's frame.

// The launch's CTAs come first in the (one-dimensional) grid, its helper CTAs after (`serve`); a
// CTA's coordinates are its index's in the launch's grid (`Ctx.gx`, `gy`).
#define PROLOGUE                                                                                 \
  extern __shared__ uint32_t lift_dyn[];                                                         \
  if (blockIdx.x >= ctx0.ctas) { lift::serve(ctx0, lift_dyn); return; }                         \
  __shared__ uint32_t s_live[LIFT_WARPS];                                                        \
  LIFT_BARS                                                                                      \
  const uint32_t tidx = threadIdx.x, tidy = threadIdx.y, tidz = threadIdx.z;                     \
  const uint32_t flat = tidx + blockDim.x * (tidy + blockDim.y * tidz);                          \
  const uint32_t lane = flat % 32, warp = flat / 32;                                             \
  const uint32_t cta = blockIdx.x;                                                               \
  const uint32_t ctaX = cta % ctx0.gx, ctaY = (cta / ctx0.gx) % ctx0.gy, ctaZ = cta / (ctx0.gx * ctx0.gy); \
  lift::Ctx ctx = ctx0;                                                                          \
  ctx.cta = cta;                                                                                 \
  ctx.stage = lift_dyn + warp * (lift::STAGE_WORDS + lift::JOB_WORDS);                           \
  ctx.shared = ctx0.sharedAll + (uint64_t)cta * ctx0.sharedBytes;                                \
  {                                                                                              \
    uint32_t _bx = ctaX - ctaX % ctx0.cx, _by = ctaY - ctaY % ctx0.cy, _bz = ctaZ - ctaZ % ctx0.cz; \
    ctx.clusterRank = ctaX % ctx0.cx + ctx0.cx * (ctaY % ctx0.cy + ctx0.cy * (ctaZ % ctx0.cz));    \
    ctx.clusterBase = _bx + ctx0.gx * (_by + ctx0.gy * _bz);                                       \
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
  __syncthreads();                                                                               \
  uint32_t act = s_live[warp];

#define EPILOGUE return;
// A thread's exit; the warp's last lane out takes the warp out of the live warps (`warpExit`).
#define LIFT_EXIT { if (atomicAnd(&LIVE, ~(1u << lane)) == (1u << lane)) lift::warpExit(ctx); return; }

// Uniform registers and predicates: each thread's own copy, written by every executing lane with
// the same value (a lane that does not execute a uniform write keeps its old value).
#define UR(i) (urf[(i)])
#define URW(i, e) do { uint32_t _v = (e); urf[(i)] = _v; } while (0)
#define UP(i) (upf[(i)])
#define UPW(i, e) do { bool _v = (e); upf[(i)] = _v; } while (0)
// B registers (`BSSY`'s lanes): each thread's own copy, so lanes that independent thread scheduling
// runs apart cannot overwrite each other's.
#define BREG(b) (breg[(b)])
// A kernel with `BREAK` (lift.py's LIFT_BREAK): its convergence barriers' state shared by the warp's
// lanes (`bsync`), per barrier the lanes arrived at its `BSYNC` and those that broke out.
#if defined(LIFT_BREAK) && LIFT_BREAK
#define LIFT_BARS __shared__ uint32_t s_bar[LIFT_WARPS][16][2];
#else
#define LIFT_BARS
#endif
#define LIVE (s_live[warp])
#define C32(off) lift::ld32c(ctx, (uint32_t)(off))
// A shared address names the CTA by its cluster rank in bits 24 and up (`Exec.sharedAt`): the
// executing CTA's own rank for its own memory.
#define SH(a) (lift::sharedAt(ctx, (uint32_t)(a)))
// A CTA's own shared window is touched by the CTA alone, on one SM: its accesses are relaxed at CTA
// scope, which the SM's L1 serves (volatile accesses would go to the L2 every time). A cluster
// peer's window (distributed shared memory) is written from another SM: accesses to it are at GPU
// scope, so a load does not read a line the L1 kept from before the peer wrote it.
#define LIFT_OWN(p) ((uint64_t)((const uint8_t*)(p) - ctx.shared) < ctx.sharedBytes)
#define LD_SH(T, PT, p) ([&]() { T _v; if (LIFT_OWN(p)) asm volatile("ld.relaxed.cta.global." PT " %0, [%1];" : "=r"(_v) : "l"(p) : "memory"); \
  else asm volatile("ld.relaxed.gpu.global." PT " %0, [%1];" : "=r"(_v) : "l"(p) : "memory"); return _v; }())
#define ST_SH(PT, p, v) do { if (LIFT_OWN(p)) asm volatile("st.relaxed.cta.global." PT " [%0], %1;" :: "l"(p), "r"(v) : "memory"); \
  else asm volatile("st.relaxed.gpu.global." PT " [%0], %1;" :: "l"(p), "r"(v) : "memory"); } while (0)
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
