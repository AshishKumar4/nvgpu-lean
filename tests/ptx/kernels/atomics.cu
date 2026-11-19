// Floating-point atomics: atomicAdd on float (global and shared), half2 and double. Each block
// also sums into a global word of its own, to compare with its shared sum.
#include <cuda_fp16.h>

extern "C" __global__ void atomics(float* gsum, const float* x, __half2* hsum, const __half2* hx,
                                   double* dsum, const double* dx, int n) {
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = 0.0f;
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(gsum, x[i]);
    atomicAdd(gsum + 4 + blockIdx.x, x[i]);
    atomicAdd(&ssum, x[i]);
    atomicAdd(hsum, hx[i]);
    atomicAdd(dsum, dx[i]);
  }
  __syncthreads();
  if (threadIdx.x == 0) gsum[1 + blockIdx.x] = ssum;
}
