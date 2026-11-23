// A pair of CTAs (a cluster of 2) runs one tcgen05.mma.cta_group::2.kind::f16 of M = 128, N = 64,
// K = 16 (B200, sm_100a; Layout B, PTX ISA §9.7.18.10.5.2): each CTA holds its 64 rows of A and its
// 32 columns of B in shared memory, in the no-swizzle K-major layout; one warp of each allocates
// 32 columns of the pair's tensor memory, the even CTA's thread 0 issues the MMA and commits it to
// both CTAs' mbarriers, and each CTA's warps load their 64 rows of D (f32) with
// tcgen05.ld.32x32b.x32: warp w reads lanes 32w to 32w + 31, rows 32 (w % 2) on, columns 32 (w / 2)
// on. Exact in f32.
#include <cuda_fp16.h>

constexpr int M = 128, N = 64, K = 16;

__device__ int at(int mn, int k) { return (mn / 8) * 256 + (k / 8) * 128 + (mn % 8) * 16 + (k % 8) * 2; }

__device__ unsigned long long sdesc(unsigned addr) {
  return (unsigned long long)((addr >> 4) & 0x3FFF) | (unsigned long long)(128 >> 4) << 16 |
         (unsigned long long)(256 >> 4) << 32 | 1ull << 46;
}

extern "C" __global__ void __cluster_dims__(2, 1, 1) tcgen05_pair128(float* out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __shared__ alignas(1024) unsigned char a[M / 2 * K * 2];
  __shared__ alignas(1024) unsigned char b[N / 2 * K * 2];
  __shared__ alignas(8) unsigned long long bar;
  __shared__ unsigned taddr;
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int t = threadIdx.x, warp = t / 32;
  for (int e = t; e < M / 2 * K; e += blockDim.x) {
    const int i = e / K, k = e % K, gi = rank * (M / 2) + i;
    *(__half*)(a + at(i, k)) = __int2half_rn((gi + k) % 5 - 2);
  }
  for (int e = t; e < N / 2 * K; e += blockDim.x) {
    const int j = e / K, k = e % K, gj = rank * (N / 2) + j;
    *(__half*)(b + at(j, k)) = __int2half_rn((3 * k + gj) % 7 - 3);
  }
  const unsigned barA = (unsigned)__cvta_generic_to_shared(&bar);
  if (warp == 0) {
    const unsigned dst = (unsigned)__cvta_generic_to_shared(&taddr);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 32;" :: "r"(dst) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
  }
  if (t == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barA) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const unsigned d = taddr;
  if (rank == 0 && t == 0) {
    const unsigned long long da = sdesc((unsigned)__cvta_generic_to_shared(a));
    const unsigned long long db = sdesc((unsigned)__cvta_generic_to_shared(b));
    const unsigned idesc = 1u << 4 | (N >> 3) << 17 | (M >> 4) << 24;
    asm volatile("{ .reg .pred p; setp.ne.u32 p, 0, 0;\n"
                 "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p; }"
                 :: "r"(d), "l"(da), "l"(db), "r"(idesc) : "memory");
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :: "r"(barA), "h"((unsigned short)3) : "memory");
  }
  unsigned done = 0;
  while (!done)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(barA) : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  unsigned r[32];
  const unsigned row = d + ((unsigned)(32 * warp) << 16);
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),"=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),"=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),"=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31]) : "r"(row));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  const int rowInCta = 32 * (warp % 2) + t % 32, col0 = 32 * (warp / 2);
  for (int j = 0; j < 32; ++j) out[(rank * 64 + rowInCta) * N + col0 + j] = __uint_as_float(r[j]);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (warp == 0) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 32;" :: "r"(d) : "memory");
  }
#endif
}
