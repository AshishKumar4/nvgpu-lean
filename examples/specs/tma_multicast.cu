// A TMA load multicast across a cluster of two CTAs: each CTA loads one 16-row tile of a 2D tensor
// into both CTAs' shared memory (`.multicast::cluster`, mask 0b11), each CTA's mbarrier counting
// both tiles' bytes, then each CTA writes out the two tiles it holds. Cluster barriers order the
// mbarriers' initialization before the copies and keep each CTA's shared memory alive until its
// peer's copy into it has landed.
#include <cuda.h>

extern "C" __global__ void __cluster_dims__(2, 1, 1) tma_multicast(const __grid_constant__ CUtensorMap map, float* out) {
  __shared__ alignas(128) float tile[2][16][32];
  __shared__ alignas(8) unsigned long long bar;
  unsigned rank;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  unsigned barA = (unsigned)__cvta_generic_to_shared(&bar);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barA) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(barA), "r"(2 * 16 * 32 * 4) : "memory");
    unsigned dst = (unsigned)__cvta_generic_to_shared(&tile[rank][0][0]);
    int x = 0, y = 16 * (int)rank + 32 * (int)(blockIdx.x / 2);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(dst), "l"(&map), "r"(x), "r"(y), "r"(barA), "h"((unsigned short)3) : "memory");
  }
  asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0; @!p bra W%=; }"
               :: "r"(barA) : "memory");
  for (int i = threadIdx.x; i < 2 * 16 * 32; i += blockDim.x) out[blockIdx.x * 1024 + i] = (&tile[0][0][0])[i];
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
