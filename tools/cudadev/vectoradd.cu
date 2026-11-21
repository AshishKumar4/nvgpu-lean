// The runtime API (cudart, as nvcc links it) on the emulated device: vectorAdd, as the CUDA
// samples have it, checked on the host.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i] + 0.0f;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

int main() {
  int n = 50000;
  size_t bytes = n * sizeof(float);
  float *a = new float[n], *b = new float[n], *c = new float[n];
  for (int i = 0; i < n; i++) { a[i] = i * 0.25f; b[i] = 1.0f / (i + 1); }
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("%s: compute %d.%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
  float *da, *db, *dc;
  CK(cudaMalloc(&da, bytes));
  CK(cudaMalloc(&db, bytes));
  CK(cudaMalloc(&dc, bytes));
  CK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
  vectorAdd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  CK(cudaGetLastError());
  CK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; i++) if (c[i] != a[i] + b[i]) bad++;
  printf("vectorAdd: %s (%d of %d wrong)\n", bad ? "FAIL" : "ok", bad, n);
  return bad != 0;
}
