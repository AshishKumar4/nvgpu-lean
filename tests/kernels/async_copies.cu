// Asynchronous copies for sm_90a, each checked exactly by Tests/AsyncKernels.lean:
// cpasync: cp.async of 16, 8 and 4 bytes (with src-size and ignore-src zero fill), completed by
//   wait_group and by an mbarrier through cp.async.mbarrier.arrive.noinc;
// bulk: cp.async.bulk global to shared (mbarrier complete-tx), an in-place update, then
//   cp.async.bulk shared to global (bulk group);
// tma_oob: a 2-D tiled TMA load whose box crosses the tensor's edges (zero fill), copied out by
//   the threads, and the same box stored back with a TMA store (out-of-bound elements unwritten).
#include <cstdint>

struct alignas(64) TensorMap { unsigned char bytes[128]; };

__device__ __forceinline__ uint32_t smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t done;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(smem(bar)), "r"(parity) : "memory");
  } while (!done);
}

extern "C" __global__ void cpasync(const unsigned char* src, unsigned char* out, int srcSize) {
  __shared__ __align__(16) unsigned char a[128 * 16];
  __shared__ __align__(16) unsigned char b[128 * 8];
  __shared__ __align__(16) unsigned char c[128 * 4];
  __shared__ uint64_t bar;
  const int t = threadIdx.x;
  if (t == 0) mbar_init(&bar, 128);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem(a + 16 * t)), "l"(src + 16 * t) : "memory");
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"
               :: "r"(smem(b + 8 * t)), "l"(src + 2048 + 8 * t), "r"(t % 9) : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("{\n .reg .pred p;\n setp.eq.u32 p, %2, 1;\n"
               " cp.async.ca.shared.global [%0], [%1], 4, p;\n}"
               :: "r"(smem(c + 4 * t)), "l"(src + 3072 + 4 * t), "r"(t % 3) : "memory");
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(smem(&bar)) : "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  mbar_wait(&bar, 0);
  __syncthreads();
  for (int i = 0; i < 16; ++i) out[16 * t + i] = a[16 * t + i];
  for (int i = 0; i < 8; ++i) out[2048 + 8 * t + i] = b[8 * t + i];
  for (int i = 0; i < 4; ++i) out[3072 + 4 * t + i] = c[4 * t + i];
}

extern "C" __global__ void bulk(const uint32_t* src, uint32_t* out) {
  __shared__ __align__(128) uint32_t buf[1024];
  __shared__ uint64_t bar;
  const int t = threadIdx.x;
  if (t == 0) {
    mbar_init(&bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (t == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4096;" :: "r"(smem(&bar)) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 4096, [%2];"
                 :: "r"(smem(buf)), "l"(src), "r"(smem(&bar)) : "memory");
  }
  mbar_wait(&bar, 0);
  for (int i = t; i < 1024; i += blockDim.x) buf[i] = buf[i] * 3 + i;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (t == 0) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 4096;"
                 :: "l"(out), "r"(smem(buf)) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
}

// Bulk reductions into global memory: out[i] = max(out[i], (int)s[i]) as s32, then
// out[1024 + i] += s[i] as u32, from 1024 shared words (cp.reduce.async.bulk, UBLKRED).
extern "C" __global__ void bulk_red(uint32_t* out) {
  __shared__ __align__(128) uint32_t s[1024];
  const int t = threadIdx.x;
  for (int i = t; i < 1024; i += blockDim.x) s[i] = i * 2654435761u;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (t == 0) {
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32 [%0], [%1], 4096;"
                 :: "l"(out), "r"(smem(s)) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], 4096;"
                 :: "l"(out + 1024), "r"(smem(s)) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
}

// Grid (2, 2): CTA (x, y) loads the 16 × 8 box at (16x - 4, 8y - 3) of a 20 × 10 f32 tensor.
extern "C" __global__ void tma_oob(const __grid_constant__ TensorMap in, const __grid_constant__ TensorMap out2,
                                   float* boxes) {
  __shared__ __align__(128) float box[8 * 16];
  __shared__ uint64_t bar;
  const int t = threadIdx.x;
  const int x0 = 16 * blockIdx.x - 4, y0 = 8 * blockIdx.y - 3;
  if (t == 0) {
    mbar_init(&bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 512;" :: "r"(smem(&bar)) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem(box)), "l"(&in), "r"(x0), "r"(y0), "r"(smem(&bar)) : "memory");
  }
  __syncthreads();
  mbar_wait(&bar, 0);
  const int cta = blockIdx.y * gridDim.x + blockIdx.x;
  boxes[128 * cta + t] = box[t];
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (t == 0) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                 :: "l"(&out2), "r"(x0), "r"(y0), "r"(smem(box)) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
}
