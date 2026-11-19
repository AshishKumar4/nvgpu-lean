// tcgen05 (sm_100a): one MMA of M = 128, N = 64, K = 16 reads A and B from shared memory, as
// examples/specs/tcgen05_gemm.cu; then thread 0 overwrites B. Ordered (WAIT): after the MMA's
// tcgen05.commit has arrived on an mbarrier and every thread waited on it. Racing: before, while
// the MMA may still read B (the restage-before-commit pattern).
#include <cuda_fp16.h>

constexpr int M = 128, N = 64, K = 16;

__device__ int at(int mn, int k) { return (mn / 8) * 256 + (k / 8) * 128 + (mn % 8) * 16 + (k % 8) * 2; }

__device__ unsigned long long sdesc(unsigned addr) {
  return (unsigned long long)((addr >> 4) & 0x3FFF) | (unsigned long long)(128 >> 4) << 16 |
         (unsigned long long)(256 >> 4) << 32 | 1ull << 46;
}

template <bool WAIT>
__device__ void restage(float* out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  __shared__ alignas(1024) unsigned char a[M * K * 2];
  __shared__ alignas(1024) unsigned char b[N * K * 2];
  __shared__ alignas(8) unsigned long long bar;
  __shared__ unsigned taddr;
  const int t = threadIdx.x, warp = t / 32;
  for (int e = t; e < M * K; e += blockDim.x) *(__half*)(a + at(e / K, e % K)) = __int2half_rn(e % 3);
  for (int e = t; e < N * K; e += blockDim.x) *(__half*)(b + at(e / K, e % K)) = __int2half_rn(e % 5);
  const unsigned barA = (unsigned)__cvta_generic_to_shared(&bar);
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
                 :: "r"((unsigned)__cvta_generic_to_shared(&taddr)) : "memory");
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
    const unsigned idesc = 1u << 4 | (N >> 3) << 17 | (M >> 4) << 24;
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }"
                 :: "r"(d), "l"(da), "l"(db), "r"(idesc) : "memory");
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(barA) : "memory");
    // Racing: restage B while the MMA may still read it.
    if (!WAIT) *(__half*)(b + at(1, 1)) = __int2half_rn(7);
  }
  unsigned done = 0;
  while (!done)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(barA) : "memory");
  // Ordered: restage B after the MMA's completion is acquired.
  if (WAIT && t == 0) *(__half*)(b + at(1, 1)) = __int2half_rn(7);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  unsigned r0;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r0) : "r"(d + ((unsigned)(32 * warp) << 16)));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  out[t] = __uint_as_float(r0);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" :: "r"(d) : "memory");
  }
#endif
}

extern "C" __global__ void tc_restage_ordered(float* out) { restage<true>(out); }
extern "C" __global__ void tc_restage_race(float* out) { restage<false>(out); }
