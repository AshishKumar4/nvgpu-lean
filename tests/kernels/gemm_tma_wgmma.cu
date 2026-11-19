// A DeepGEMM-style BF16 GEMM tile kernel for sm_90a: C = A · Bᵀ in FP32, A (M×K) and B (N×K)
// both K-major in global memory. One producer warp issues TMA loads of 64×64 A tiles and 128×64
// B tiles (128-byte swizzle) into a four-stage shared-memory ring guarded by full/empty mbarriers;
// one consumer warpgroup runs wgmma m64n128k16 on each stage (four per 64-wide K block) and
// writes its 64×128 FP32 tile of C. Grid: (N / 128, M / 64), 256 threads per CTA.
#include <cstdint>

struct alignas(64) TensorMap { unsigned char bytes[128]; };

constexpr int BM = 64, BN = 128, BK = 64, STAGES = 4;
constexpr int A_BYTES = BM * BK * 2, B_BYTES = BN * BK * 2;

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

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem(bar)) : "memory");
}

__device__ __forceinline__ void mbar_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tma_load(void* dst, const TensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3}], [%4];"
               :: "r"(smem(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem(bar)) : "memory");
}

// K-major operand with 128-byte swizzle: rows of 128 bytes, 8-row groups 1024 bytes apart.
__device__ __forceinline__ uint64_t desc(const void* p) {
  return ((smem(p) & 0x3FFFF) >> 4) | (1ull << 16) | ((1024ull >> 4) << 32) | (1ull << 62);
}

__device__ __forceinline__ void wgmma(float* d, uint64_t a, uint64_t b) {
  asm volatile("{\n wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 0;\n}"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b));
}

extern "C" __global__ void __launch_bounds__(256)
gemm(const __grid_constant__ TensorMap mapA, const __grid_constant__ TensorMap mapB, float* c,
     int n, int k) {
  extern __shared__ __align__(1024) unsigned char buf[];
  unsigned char* as = buf;
  unsigned char* bs = buf + STAGES * A_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(bs + STAGES * B_BYTES);
  uint64_t* empty = full + STAGES;
  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int kblocks = k / BK;
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 128) {
    if (tid == 0) {
      for (int kb = 0; kb < kblocks; ++kb) {
        const int s = kb % STAGES;
        mbar_wait(&empty[s], ((kb / STAGES) & 1) ^ 1);
        mbar_expect(&full[s], A_BYTES + B_BYTES);
        tma_load(as + s * A_BYTES, &mapA, kb * BK, m0, &full[s]);
        tma_load(bs + s * B_BYTES, &mapB, kb * BK, n0, &full[s]);
      }
    }
    return;
  }
  float d[64];
  for (int i = 0; i < 64; ++i) d[i] = 0.0f;
  for (int kb = 0; kb < kblocks; ++kb) {
    const int s = kb % STAGES;
    mbar_wait(&full[s], (kb / STAGES) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    for (int j = 0; j < BK / 16; ++j)
      wgmma(d, desc(as + s * A_BYTES + 32 * j), desc(bs + s * B_BYTES + 32 * j));
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if (tid == 128) mbar_arrive(&empty[s]);
  }
  // The accumulator fragment: value v of lane l in warp w is row 16w + l/4 + 8((v/2)%2), column
  // 2(l%4) + v%2 + 8(v/4) of the tile (PTX ISA §9.7.17.5.1.1, Figure 152).
  const int w = (tid - 128) / 32, l = tid % 32;
  for (int v = 0; v < 64; ++v) {
    const int row = 16 * w + l / 4 + 8 * ((v / 2) % 2);
    const int col = 2 * (l % 4) + v % 2 + 8 * (v / 4);
    c[(m0 + row) * n + n0 + col] = d[v];
  }
}
