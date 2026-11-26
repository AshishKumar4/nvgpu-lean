// A bulk copy multicast across a cluster of two CTAs: each CTA copies 2 KB of contiguous memory
// into both CTAs' shared memory (`cp.async.bulk ... .multicast::cluster`, mask 0b11, SASS
// UBLKCP.S.G.MULTICAST), each CTA's mbarrier counting both halves, then each CTA writes out the
// 4 KB it holds. Cluster barriers order the mbarriers' initialization before the copies and keep
// each CTA's shared memory alive until its peer's copy into it has landed.
extern "C" __global__ void __cluster_dims__(2, 1, 1) bulk_multicast(const unsigned* in, unsigned* out) {
  __shared__ alignas(128) unsigned buf[2][512];
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
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(barA), "r"(2 * 2048) : "memory");
    unsigned dst = (unsigned)__cvta_generic_to_shared(&buf[rank][0]);
    const unsigned* src = in + (blockIdx.x / 2) * 1024 + rank * 512;
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], 2048, [%2], %3;"
        :: "r"(dst), "l"(src), "r"(barA), "h"((unsigned short)3) : "memory");
  }
  asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0; @!p bra W%=; }"
               :: "r"(barA) : "memory");
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) out[blockIdx.x * 1024 + i] = (&buf[0][0])[i];
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
