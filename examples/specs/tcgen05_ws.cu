// tcgen05.mma.ws.cta_group::1.kind::f16 (weight stationary, B200, sm_100a) of M = 32, 64 and 128,
// N = 64, K = 16, from the same A (128 × 16) and B (16 × 64) of small integers as f16 in shared
// memory (no-swizzle K-major): D of M = 32 at column 0 (Layout G: a quarter of the columns in each
// lane quarter, 16 columns), of M = 64 at column 32 (Layout E: the halves in lanes 0–63 and 64–127,
// 32 columns), of M = 128 at column 64 (Layout D). The second carries a collector hint, the third
// an empty zero-column mask. Each warp then loads its 32 lanes of all 128 columns, which go to
// `out` lane by lane. Every product and sum is exact in f32.
#include <cuda_fp16.h>

constexpr int M = 128, N = 64, K = 16;

// Element (mn, k) of a K-major operand without swizzling: 8-row by 16-byte core matrices, the
// two along K 128 bytes apart (LBO), row groups 256 bytes apart (SBO).
__device__ int at(int mn, int k) { return (mn / 8) * 256 + (k / 8) * 128 + (mn % 8) * 16 + (k % 8) * 2; }

__device__ unsigned long long sdesc(unsigned addr) {
  return (unsigned long long)((addr >> 4) & 0x3FFF) | (unsigned long long)(128 >> 4) << 16 |
         (unsigned long long)(256 >> 4) << 32 | 1ull << 46;
}

extern "C" __global__ void tcgen05_ws(float* out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __shared__ alignas(1024) unsigned char a[128 * K * 2];
  __shared__ alignas(1024) unsigned char b[N * K * 2];
  __shared__ alignas(8) unsigned long long bar;
  __shared__ unsigned taddr;
  const int t = threadIdx.x, warp = t / 32;
  for (int e = t; e < 128 * K; e += blockDim.x) {
    const int i = e / K, k = e % K;
    *(__half*)(a + at(i, k)) = __int2half_rn((i + k) % 5 - 2);
  }
  for (int e = t; e < N * K; e += blockDim.x) {
    const int j = e / K, k = e % K;
    *(__half*)(b + at(j, k)) = __int2half_rn((3 * k + j) % 7 - 3);
  }
  const unsigned barA = (unsigned)__cvta_generic_to_shared(&bar);
  if (warp == 0) {
    const unsigned dst = (unsigned)__cvta_generic_to_shared(&taddr);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" :: "r"(dst) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (t == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barA) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const unsigned d = taddr;
  if (t == 0) {
    const unsigned long long da = sdesc((unsigned)__cvta_generic_to_shared(a));
    const unsigned long long db = sdesc((unsigned)__cvta_generic_to_shared(b));
    // D f32 (bit 4), A and B f16, K-major, N >> 3 at bit 17, M >> 4 at bit 24.
    const unsigned i32 = 1u << 4 | (N >> 3) << 17 | (32 >> 4) << 24;
    const unsigned i64 = 1u << 4 | (N >> 3) << 17 | (64 >> 4) << 24;
    const unsigned i128 = 1u << 4 | (N >> 3) << 17 | (128 >> 4) << 24;
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }"
                 :: "r"(d), "l"(da), "l"(db), "r"(i32) : "memory");
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.ws.cta_group::1.kind::f16.collector::b0::fill [%0], %1, %2, %3, p; }"
                 :: "r"(d + 32), "l"(da), "l"(db), "r"(i64) : "memory");
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, %4; }"
                 :: "r"(d + 64), "l"(da), "l"(db), "r"(i128), "l"(0ull) : "memory");
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(barA) : "memory");
  }
  unsigned done = 0;
  while (!done)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(barA) : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  unsigned r[64];
  const unsigned row = d + ((unsigned)(32 * warp) << 16);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
               "%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,"
               "%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
               : "r"(row + 0u));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  for (int j = 0; j < 64; ++j) out[t * 128 + 0 + j] = __uint_as_float(r[j]);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
               "%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,"
               "%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]), "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
               : "r"(row + 64u));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  for (int j = 0; j < 64; ++j) out[t * 128 + 64 + j] = __uint_as_float(r[j]);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(d) : "memory");
  }
#endif
}
