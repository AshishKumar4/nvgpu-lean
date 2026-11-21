// Warpgroup 0 runs wgmma m64n128k16 bf16 (A and B from shared memory) back to back while warpgroup 1, on the same
// sub-partitions, walks a dependent chain of shared-memory loads; the timing tests read the stall counters.
typedef unsigned long long u64;

__device__ __forceinline__ void mma_ss(float* d, u64 a, u64 b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
    "%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, "
    "%47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, 0, 0;\n}"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
      "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
      "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
      "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
      "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
      "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
      "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
      "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
    : "l"(a), "l"(b));
#endif
}

// A K-major operand descriptor with the 128-byte swizzle.
__device__ __forceinline__ u64 desc(unsigned addr) {
  return (u64)((addr >> 4) & 0x3fff) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

extern "C" __global__ void beside(unsigned* out, int rounds, int steps) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const int tid = threadIdx.x, wg = __shfl_sync(0xffffffffu, tid / 128, 0);
  const unsigned base = (unsigned)__cvta_generic_to_shared(smem);
  for (int i = tid; i < 8192 / 4 + 16384 / 4; i += blockDim.x) ((unsigned*)smem)[i] = 0x3f803f80u;
  // The chain: word i of the last 4 KB holds the offset of word (i * 33 + 1) % 1024.
  for (int i = tid; i < 1024; i += blockDim.x) ((unsigned*)(smem + 24576))[i] = 24576 + ((i * 33 + 1) % 1024) * 4;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (wg == 0) {
    float d[64];
    for (int i = 0; i < 64; ++i) d[i] = 0.f;
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
    for (int r = 0; r < rounds; ++r) {
#pragma unroll
      for (int q = 0; q < 4; ++q) mma_ss(d, desc(base + 32 * q), desc(base + 8192 + 32 * q));
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
#endif
    }
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#endif
    float s = 0.f;
    for (int i = 0; i < 64; ++i) s += d[i];
    if (tid == 0) out[0] = __float_as_uint(s);
  } else {
    unsigned p = 24576 + (tid % 32) * 4;
    for (int i = 0; i < steps; ++i) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(p) : "r"(base + p));
    if (tid == 128) out[1] = p;
  }
}

// A warpgroup commits a wgmma group with nothing in it and waits for it (ptxas fills the group with
// an empty `HGMMA.64x8x16.F16 RZ, gdesc[URZ], RZ, !UPT`), then writes its thread indices.
extern "C" __global__ void empty_group(unsigned* out) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#endif
  out[threadIdx.x] = threadIdx.x;
}
