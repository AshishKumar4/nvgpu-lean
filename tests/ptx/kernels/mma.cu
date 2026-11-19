// One m16n8k16 tile, D = A·B + C: A (16x16, row-major) and Bᵀ (8x16, row-major) staged in shared
// memory and loaded with ldmatrix, multiplied with mma.sync in f32. Then A is stored back with
// stmatrix (a round trip) and its first 8x8 block transposed with movmatrix.
#include <cuda_fp16.h>

extern "C" __global__ void mma_m16n8k16(const __half* a, const __half* bt, const float* c, float* d,
                                        __half* a_back, unsigned* a0_t) {
  __shared__ __align__(16) __half sa[16 * 16];
  __shared__ __align__(16) __half sb[8 * 16];
  __shared__ __align__(16) __half sback[16 * 16];
  int l = threadIdx.x;
  for (int i = l; i < 256; i += 32) sa[i] = a[i];
  for (int i = l; i < 128; i += 32) sb[i] = bt[i];
  __syncwarp();
  unsigned a0, a1, a2, a3, b0, b1;
  unsigned pa = (unsigned)__cvta_generic_to_shared(&sa[(l % 16) * 16 + (l / 16) * 8]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(pa));
  unsigned pb = (unsigned)__cvta_generic_to_shared(&sb[(l % 8) * 16 + ((l / 8) % 2) * 8]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(b0), "=r"(b1) : "r"(pb));
  int g = l / 4, q = l % 4;
  float c0 = c[g * 8 + q * 2], c1 = c[g * 8 + q * 2 + 1];
  float c2 = c[(g + 8) * 8 + q * 2], c3 = c[(g + 8) * 8 + q * 2 + 1];
  float d0, d1, d2, d3;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  d[g * 8 + q * 2] = d0;
  d[g * 8 + q * 2 + 1] = d1;
  d[(g + 8) * 8 + q * 2] = d2;
  d[(g + 8) * 8 + q * 2 + 1] = d3;
  unsigned ps = (unsigned)__cvta_generic_to_shared(&sback[(l % 16) * 16 + (l / 16) * 8]);
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
               :: "r"(ps), "r"(a0), "r"(a1), "r"(a2), "r"(a3) : "memory");
  __syncwarp();
  for (int i = l; i < 256; i += 32) a_back[i] = sback[i];
  unsigned t;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(t) : "r"(a0));
  a0_t[l] = t;
}
