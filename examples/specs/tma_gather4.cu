// Row gathers and scatters with the tensor memory accelerator (sm_100a): each CTA loads four rows
// of a 64 x 32 f32 tensor, chosen by its index (`cp.async.bulk.tensor.2d ... .tile::gather4`,
// SASS UTMALDG.2D.GATHER4) under the 128-byte swizzle, then stores them to four other rows of a
// second tensor (`.tile::scatter4`, UTMASTG.2D.SCATTER4), and writes the shared tile out as it is.
#include <cuda.h>

extern "C" __global__ void tma_gather4(const __grid_constant__ CUtensorMap src, const __grid_constant__ CUtensorMap dst,
                                       unsigned* out) {
  __shared__ alignas(1024) unsigned tile[4][32];
  __shared__ alignas(8) unsigned long long bar;
  const unsigned barA = (unsigned)__cvta_generic_to_shared(&bar), t = (unsigned)__cvta_generic_to_shared(tile);
  const int b = blockIdx.x;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barA) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(barA), "r"(512) : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(t), "l"(&src), "r"(0), "r"(b * 7 % 64), "r"((b * 13 + 5) % 64), "r"((b * 3 + 40) % 64), "r"(63 - b),
           "r"(barA) : "memory");
  }
  asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0; @!p bra W%=; }"
               :: "r"(barA) : "memory");
  if (threadIdx.x == 0) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
        :: "l"(&dst), "r"(0), "r"(4 * b), "r"(4 * b + 2), "r"(4 * b + 1), "r"(4 * b + 3), "r"(t) : "memory");
    asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group.read 0;" ::: "memory");
  }
  for (int i = threadIdx.x; i < 128; i += blockDim.x) out[b * 128 + i] = (&tile[0][0])[i];
  __syncthreads();
}
