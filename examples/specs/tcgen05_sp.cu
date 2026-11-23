// tcgen05.mma.sp.cta_group::1 (sparse A, B200, sm_100a), from shared memory without swizzling
// (K-major), each with its sparse metadata in tensor memory (PTX ISA §9.7.18.10.9): .kind::f16 of
// M = 128 (D at column 0; the metadata's fields in column 257, the selector 1 picking it past 256,
// whose word would pick other elements), of M = 64 (D at column 64, metadata 260; Layout F, the
// first 16 lanes of each quarter), .kind::tf32 of M = 128 (D at 128, metadata 264), .kind::i8 of
// M = 128 (D in S32 at 192, metadata 268 and 269), N = 64. A's stored elements and B are small
// integers, the metadata's fields a varying choice of places, so every sum is exact; out holds
// D's 256 columns of each lane (zero where M = 64 has no row).
#include <cuda_fp16.h>

constexpr int N = 64;

// Element (mn, k) of a K-major operand without swizzling, `bytes` an element, `cores` 16-byte core
// matrices along K: 8-row by 16-byte core matrices, 128 bytes apart along K (LBO), row groups
// 128 * cores apart (SBO).
__device__ int at(int mn, int k, int bytes, int cores) {
  const int per = 16 / bytes;
  return (mn / 8) * 128 * cores + (k / per) * 128 + (mn % 8) * 16 + (k % per) * bytes;
}

__device__ unsigned long long sdesc(unsigned addr, int cores) {
  return (unsigned long long)((addr >> 4) & 0x3FFF) | (unsigned long long)(128 >> 4) << 16 |
         (unsigned long long)((128 * cores) >> 4) << 32 | 1ull << 46;
}

// The metadata field of row r's chunk c: two places of four (bits 0-1 the first stored element's,
// 2-3 the second's), or under tf32 which of two.
__device__ unsigned field2of4(int v) {
  const unsigned places[6] = {0x4, 0x8, 0xC, 0x9, 0xD, 0xE};
  return places[v % 6];
}
__device__ unsigned f16Field(int r, int c) { return field2of4(3 * r + 5 * c); }
__device__ unsigned i8Field(int r, int c) { return field2of4(r + 2 * c); }
__device__ unsigned tf32Field(int r, int c) { return (r + c) % 2 ? 0xE : 0x4; }

// The metadata word of lane `l` of a 16-bit kind's layout whose 16-row groups start every `step`
// lanes: rows 16 g + 8 h + l % 8 at bits 16 h, chunks 4 ((l % step) / 8) + j at bits 4 j.
template <class F>
__device__ unsigned wideWord(int l, int step, F field) {
  if (l % step >= 16) return 0;
  const int g = l / step, hi = (l % step) / 8;
  unsigned w = 0;
  for (int h = 0; h < 2; ++h)
    for (int j = 0; j < 4; ++j) w |= field(16 * g + 8 * h + l % 8, 4 * hi + j) << (16 * h + 4 * j);
  return w;
}

__device__ void st1(unsigned taddr, unsigned v) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(taddr), "r"(v) : "memory");
}

extern "C" __global__ void tcgen05_sp(unsigned* out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  // A: 128 rows of 32 bytes stored (16 f16, 8 tf32, 32 s8); B: 64 rows of 64 bytes.
  __shared__ alignas(1024) unsigned char ah[128 * 32], bh[N * 64];
  __shared__ alignas(1024) unsigned char at32[128 * 32], bt32[N * 64];
  __shared__ alignas(1024) unsigned char ai[128 * 32], bi[N * 64];
  __shared__ alignas(8) unsigned long long bar;
  __shared__ unsigned taddr;
  const int t = threadIdx.x, warp = t / 32;
  for (int e = t; e < 128 * 32; e += blockDim.x) {
    const int r = e / 32, s = e % 32, v = (r + 2 * s) % 5 - 2;
    if (s < 16) *(__half*)(ah + at(r, s, 2, 2)) = __int2half_rn(v);
    if (s < 8) *(float*)(at32 + at(r, s, 4, 2)) = (float)v;
    *(signed char*)(ai + at(r, s, 1, 2)) = (signed char)v;
  }
  for (int e = t; e < N * 64; e += blockDim.x) {
    const int n = e / 64, k = e % 64, v = (3 * k + n) % 7 - 3;
    if (k < 32) *(__half*)(bh + at(n, k, 2, 4)) = __int2half_rn(v);
    if (k < 16) *(float*)(bt32 + at(n, k, 4, 4)) = (float)v;
    *(signed char*)(bi + at(n, k, 1, 4)) = (signed char)v;
  }
  const unsigned barA = (unsigned)__cvta_generic_to_shared(&bar);
  if (warp == 0) {
    const unsigned dst = (unsigned)__cvta_generic_to_shared(&taddr);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(dst) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (t == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barA) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const unsigned d = taddr;
  // The metadata, each warp its 32 lanes.
  const unsigned lanes = d + ((unsigned)(32 * warp) << 16);
  st1(lanes + 256, 0x44444444u);
  st1(lanes + 257, wideWord(t, 16, f16Field));
  st1(lanes + 260, wideWord(t, 32, f16Field));
  st1(lanes + 264, wideWord(t, 16, tf32Field));
  unsigned w0 = 0, w1 = 0;
  for (int j = 0; j < 8; ++j) { w0 |= i8Field(t, j) << (4 * j); w1 |= i8Field(t, 8 + j) << (4 * j); }
  st1(lanes + 268, w0);
  st1(lanes + 269, w1);
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (t == 0) {
    const unsigned long long dah = sdesc((unsigned)__cvta_generic_to_shared(ah), 2);
    const unsigned long long dbh = sdesc((unsigned)__cvta_generic_to_shared(bh), 4);
    const unsigned long long dat = sdesc((unsigned)__cvta_generic_to_shared(at32), 2);
    const unsigned long long dbt = sdesc((unsigned)__cvta_generic_to_shared(bt32), 4);
    const unsigned long long dai = sdesc((unsigned)__cvta_generic_to_shared(ai), 2);
    const unsigned long long dbi = sdesc((unsigned)__cvta_generic_to_shared(bi), 4);
    // The selector in bits 0-1, sparse (bit 2), D's type at bit 4 (1 F32, 2 S32), A's and B's at
    // bits 7 and 10 (f16 0, tf32 2, s8 1), N >> 3 at bit 17, M >> 4 at bit 24.
    const unsigned h128 = 1u | 1u << 2 | 1u << 4 | (N >> 3) << 17 | (128 >> 4) << 24;
    const unsigned h64 = 1u << 2 | 1u << 4 | (N >> 3) << 17 | (64 >> 4) << 24;
    const unsigned t128 = 1u << 2 | 1u << 4 | 2u << 7 | 2u << 10 | (N >> 3) << 17 | (128 >> 4) << 24;
    const unsigned i128 = 1u << 2 | 2u << 4 | 1u << 7 | 1u << 10 | (N >> 3) << 17 | (128 >> 4) << 24;
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.sp.cta_group::1.kind::f16 [%0], %1, %2, [%3], %4, p; }"
                 :: "r"(d), "l"(dah), "l"(dbh), "r"(d + 256), "r"(h128) : "memory");
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.sp.cta_group::1.kind::f16 [%0], %1, %2, [%3], %4, p; }"
                 :: "r"(d + 64), "l"(dah), "l"(dbh), "r"(d + 260), "r"(h64) : "memory");
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.sp.cta_group::1.kind::tf32 [%0], %1, %2, [%3], %4, p; }"
                 :: "r"(d + 128), "l"(dat), "l"(dbt), "r"(d + 264), "r"(t128) : "memory");
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.sp.cta_group::1.kind::i8 [%0], %1, %2, [%3], %4, p; }"
                 :: "r"(d + 192), "l"(dai), "l"(dbi), "r"(d + 268), "r"(i128) : "memory");
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(barA) : "memory");
  }
  unsigned done = 0;
  while (!done)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(barA) : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  unsigned r[64];
  const unsigned row = lanes;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
               : "r"(row + 0u));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  for (int j = 0; j < 64; ++j) out[t * 256 + 0 + j] = r[j];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
               : "r"(row + 64u));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  for (int j = 0; j < 64; ++j) out[t * 256 + 64 + j] = t % 32 < 16 ? r[j] : 0u;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
               : "r"(row + 128u));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  for (int j = 0; j < 64; ++j) out[t * 256 + 128 + j] = r[j];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
               : "r"(row + 192u));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  for (int j = 0; j < 64; ++j) out[t * 256 + 192 + j] = r[j];
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(d) : "memory");
  }
#endif
}
