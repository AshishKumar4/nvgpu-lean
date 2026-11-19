#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void fimm(float* o, const float* a, double* d, __half2* h, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[i];
  float y = x * 0.1f + 3.0f;
  y = fmaf(y, x, 0.5f);
  y = y + 1e30f;
  y = (x > 0.f) ? y : -INFINITY;
  y = isnan(x) ? __int_as_float(0x7fffffff) : y;
  y = fminf(y, -0.0f);
  o[i] = y;
  double z = d[i] * 0.1 + 2.0;
  d[i] = z / 2147482624.0;
  __half2 hv = h[i];
  hv = __hfma2(hv, __floats2half2_rn(0.5f, 1.5f), __floats2half2_rn(-2.0f, 0.25f));
  h[i] = hv;
}
extern "C" __global__ void mathk(float* o, const float* a, unsigned* u) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[i];
  o[i] = __expf(x) + __sinf(x) + __logf(x) + rsqrtf(x) + sqrtf(x) + 1.0f / x + exp2f(x) + __fdividef(x, x + 1.f);
  unsigned v = u[i];
  u[i] = __popc(v) + __brev(v) + __clz(v) + __ffs(v) + __byte_perm(v, i, 0x5410) + (unsigned)(int)x + (unsigned)__float2int_rn(x * 3.f);
  o[i + 1] = (float)v + (float)(int)v + __int2float_rz(i);
}
extern "C" __global__ void warpk(int* o, const int* a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int v = a[i];
  int s = v;
  for (int d = 16; d > 0; d >>= 1) s += __shfl_down_sync(0xffffffff, s, d);
  unsigned b = __ballot_sync(0xffffffff, v > 0);
  int any = __any_sync(0xffffffff, v == 3);
  unsigned m = __match_any_sync(0xffffffff, v);
  int r = __reduce_add_sync(0xffffffff, v);
  o[i] = s + b + any + m + r + __shfl_sync(0xffffffff, v, 3) + __shfl_xor_sync(0xffffffff, v, 1) + __shfl_up_sync(0xffffffff, v, 2);
}
extern "C" __global__ void divk(int* o, const int* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int v = a[i];
  int acc = 0;
  if (v & 1) {
    for (int k = 0; k < v; ++k) acc += k * v;
    __syncwarp();
  } else {
    acc = v * 7;
  }
  o[i] = acc;
}
extern "C" __global__ void atomk(int* o, float* f, unsigned long long* q, int n) {
  __shared__ int s[64];
  int i = threadIdx.x;
  if (i < 64) s[i] = 0;
  __syncthreads();
  atomicAdd(&s[i % 64], i);
  atomicMax(&s[(i * 7) % 64], i);
  __syncthreads();
  int old = atomicCAS(&o[i % 8], i, i + 1);
  atomicAdd(&f[i % 4], 1.5f);
  atomicExch(&q[i % 2], (unsigned long long)i);
  atomicAdd(&o[8], s[i % 64] + old);
}
__device__ __noinline__ int helper(int x, int y) { return x * y + (x ^ y); }
extern "C" __global__ void callk(int* o, const int* a) {
  int i = threadIdx.x;
  o[i] = helper(a[i], i);
}
extern "C" __global__ void spillk(float* o, const float* a) {
  float v[64];
  int i = threadIdx.x;
  #pragma unroll
  for (int k = 0; k < 64; ++k) v[k] = a[i + k * 32];
  float s = 0;
  for (int k = 0; k < 64; ++k) s += v[(k * 13 + i) & 63];
  o[i] = s;
}
extern "C" __global__ void mmak(float* o, const unsigned* a) {
  int i = threadIdx.x;
  unsigned a0 = a[i], a1 = a[i+32], a2 = a[i+64], a3 = a[i+96], b0 = a[i+128], b1 = a[i+160];
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  unsigned r0, r1, r2, r3;
  __shared__ unsigned sm[512];
  sm[i] = a0; __syncthreads();
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"((unsigned)__cvta_generic_to_shared(&sm[(i % 32) * 4])));
  o[i] = c0 + c1 + c2 + c3 + r0 + r1 + r2 + r3;
}
extern "C" __global__ void __launch_bounds__(256, 2) bounded(float* o) { o[threadIdx.x] = 1.f; }
extern "C" __global__ void __cluster_dims__(2, 1, 1) clusterk(int* o) {
  unsigned r; asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); o[blockIdx.x * blockDim.x + threadIdx.x] = r;
}
extern "C" __global__ void dynsmem(float* o) { extern __shared__ float d[]; d[threadIdx.x] = threadIdx.x; __syncthreads(); o[threadIdx.x] = d[(threadIdx.x + 1) % blockDim.x]; }
