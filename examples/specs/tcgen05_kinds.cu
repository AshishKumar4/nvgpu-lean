// tcgen05.mma.cta_group::1 of the kinds other than f16's (B200, sm_100a), from shared memory
// without swizzling (K-major), M = 128, N = 64: .kind::tf32 (K = 8; D at column 0),
// .kind::f8f6f4 of E4M3 A and E5M2 B (K = 32; D at 64), .kind::i8 of s8 A and u8 B (K = 32; D in
// S32 at 128); and .kind::f16 with A in tensor memory (K = 16; A at column 256, row r in lane r,
// two halves a column, low first; D at 192). A and B are small integers, so every sum is exact;
// out holds D's 256 columns of each lane.
#include <cuda_fp16.h>
constexpr int N = 64;

// Element (mn, k) of a K-major operand without swizzling, `bytes` an element: 8-row by 16-byte
// core matrices, the two along K 128 bytes apart (LBO), row groups 256 bytes apart (SBO).
__device__ int at(int mn, int k, int bytes) {
  const int per = 16 / bytes;
  return (mn / 8) * 256 + (k / per) * 128 + (mn % 8) * 16 + (k % per) * bytes;
}

__device__ unsigned long long sdesc(unsigned addr) {
  return (unsigned long long)((addr >> 4) & 0x3FFF) | (unsigned long long)(128 >> 4) << 16 |
         (unsigned long long)(256 >> 4) << 32 | 1ull << 46;
}

// -3..3 as E4M3 and as E5M2.
__device__ unsigned char e4m3(int v) {
  const unsigned char m[4] = {0x00, 0x38, 0x40, 0x44};
  return (v < 0 ? 0x80 : 0) | m[v < 0 ? -v : v];
}
__device__ unsigned char e5m2(int v) {
  const unsigned char m[4] = {0x00, 0x3C, 0x40, 0x42};
  return (v < 0 ? 0x80 : 0) | m[v < 0 ? -v : v];
}

extern "C" __global__ void tcgen05_kinds(unsigned* out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  // Each operand 32 bytes a row: 8 tf32, 32 FP8 or 32 8-bit integers.
  __shared__ alignas(1024) unsigned char at32[128 * 32], bt32[N * 32], a8[128 * 32], b8[N * 32], ai[128 * 32], bi[N * 32];
  __shared__ alignas(1024) unsigned char bh[N * 32];
  __shared__ alignas(8) unsigned long long bar;
  __shared__ unsigned taddr;
  const int t = threadIdx.x, warp = t / 32;
  for (int e = t; e < 128 * 32; e += blockDim.x) {
    const int i = e / 32, k = e % 32, v = (i + 2 * k) % 5 - 2;
    if (k < 8) *(float*)(at32 + at(i, k, 4)) = (float)v;
    a8[at(i, k, 1)] = e4m3(v);
    ai[at(i, k, 1)] = (unsigned char)(signed char)v;
  }
  for (int e = t; e < N * 32; e += blockDim.x) {
    const int j = e / 32, k = e % 32, v = (3 * k + j) % 7 - 3;
    if (k < 8) *(float*)(bt32 + at(j, k, 4)) = (float)v;
    b8[at(j, k, 1)] = e5m2(v);
    bi[at(j, k, 1)] = (unsigned char)(v + 3);
    if (k < 16) *(__half*)(bh + at(j, k, 2)) = __int2half_rn(v);
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
  // A of the f16 MMA into tensor memory: this lane's row, K = 16 halves in columns 256-263.
  const unsigned lanes = d + ((unsigned)(32 * warp) << 16);
  for (int c = 0; c < 8; ++c) {
    const __half lo = __int2half_rn((t + 4 * c) % 5 - 2), hi = __int2half_rn((t + 4 * c + 2) % 5 - 2);
    const unsigned w = (unsigned)__half_as_ushort(lo) | (unsigned)__half_as_ushort(hi) << 16;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(lanes + 256 + c), "r"(w) : "memory");
  }
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (t == 0) {
    const unsigned long long dat = sdesc((unsigned)__cvta_generic_to_shared(at32)), dbt = sdesc((unsigned)__cvta_generic_to_shared(bt32));
    const unsigned long long da8 = sdesc((unsigned)__cvta_generic_to_shared(a8)), db8 = sdesc((unsigned)__cvta_generic_to_shared(b8));
    const unsigned long long dai = sdesc((unsigned)__cvta_generic_to_shared(ai)), dbi = sdesc((unsigned)__cvta_generic_to_shared(bi));
    // D's type at bit 4 (1 F32, 2 S32), A's and B's at bits 7 and 10 (tf32 2; E4M3 0, E5M2 1; s8 1,
    // u8 0), N >> 3 at bit 17, M >> 4 at bit 24.
    const unsigned tf = 1u << 4 | 2u << 7 | 2u << 10 | (N >> 3) << 17 | (128 >> 4) << 24;
    const unsigned f8 = 1u << 4 | 0u << 7 | 1u << 10 | (N >> 3) << 17 | (128 >> 4) << 24;
    const unsigned i8 = 2u << 4 | 1u << 7 | 0u << 10 | (N >> 3) << 17 | (128 >> 4) << 24;
    const unsigned h = 1u << 4 | (N >> 3) << 17 | (128 >> 4) << 24;
    const unsigned long long dbh = sdesc((unsigned)__cvta_generic_to_shared(bh));
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
                 :: "r"(d + 0), "l"(dat), "l"(dbt), "r"(tf) : "memory");
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p; }"
                 :: "r"(d + 64), "l"(da8), "l"(db8), "r"(f8) : "memory");
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, p; }"
                 :: "r"(d + 128), "l"(dai), "l"(dbi), "r"(i8) : "memory");
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p; }"
                 :: "r"(d + 192), "r"(d + 256), "l"(dbh), "r"(h) : "memory");
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
  for (int j = 0; j < 64; ++j) out[t * 256 + 64 + j] = r[j];
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
