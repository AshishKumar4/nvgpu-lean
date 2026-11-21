// The runtime API (cudart, as nvcc links it) on the emulated device, as CUDA programs use it,
// each part checked on the host: the program's fat binary registered by cudart and its kernels
// launched by name; `__constant__` and `__device__` variables through cudaMemcpyToSymbol and
// cudaMemcpyFromSymbol; streams and events; a CUDA graph captured from a stream and replayed;
// cudaLaunchKernelEx with a cluster shape on parts with clusters; occupancy.
#include <cstdio>
#include <cuda_runtime.h>

__constant__ float scale;
__device__ unsigned counter;

__global__ void scaled(const float *a, float *b, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) b[i] = a[i] * scale;
}

__global__ void bump() { atomicAdd(&counter, 1u); }

static int failures;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); return 1; } } while (0)

static void check(const char *what, bool ok) {
  printf("%s: %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) failures++;
}

int main() {
  int n = 4096;
  size_t bytes = n * sizeof(float);
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("%s: compute %d.%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
  float *a = new float[n], *b = new float[n];
  for (int i = 0; i < n; i++) a[i] = i * 0.25f;
  float *da, *db;
  CK(cudaMalloc(&da, bytes));
  CK(cudaMalloc(&db, bytes));
  CK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  float three = 3.0f;
  CK(cudaMemcpyToSymbol(scale, &three, sizeof three));
  scaled<<<(n + 255) / 256, 256>>>(da, db, n);
  CK(cudaGetLastError());
  CK(cudaMemcpy(b, db, bytes, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; i++) if (b[i] != a[i] * 3.0f) bad++;
  check("kernel reading a __constant__ the host wrote", bad == 0);

  unsigned zero = 0, got = 0;
  CK(cudaMemcpyToSymbol(counter, &zero, sizeof zero));
  bump<<<8, 128>>>();
  CK(cudaMemcpyFromSymbol(&got, counter, sizeof got));
  check("__device__ counter read back by symbol", got == 1024);

  // A stream, its events, and a graph captured from it.
  cudaStream_t s;
  cudaEvent_t e0, e1;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaMemcpyToSymbolAsync(counter, &zero, sizeof zero, 0, cudaMemcpyHostToDevice, s));
  CK(cudaStreamSynchronize(s));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  bump<<<2, 64, 0, s>>>();
  scaled<<<(n + 255) / 256, 256, 0, s>>>(da, db, n);
  bump<<<1, 32, 0, s>>>();
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaEventRecord(e0, s));
  for (int i = 0; i < 4; i++) CK(cudaGraphLaunch(ge, s));
  CK(cudaEventRecord(e1, s));
  CK(cudaEventSynchronize(e1));
  CK(cudaMemcpyFromSymbol(&got, counter, sizeof got));
  check("captured graph replayed four times", got == 4 * 160);
  float ms = -1;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  check("elapsed time", ms >= 0);

  if (p.major >= 9) {
    cudaLaunchConfig_t lc = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 2;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    lc.gridDim = dim3(8);
    lc.blockDim = dim3(32);
    lc.stream = s;
    lc.attrs = at;
    lc.numAttrs = 1;
    CK(cudaMemcpyToSymbol(counter, &zero, sizeof zero));
    CK(cudaLaunchKernelEx(&lc, bump));
    CK(cudaStreamSynchronize(s));
    CK(cudaMemcpyFromSymbol(&got, counter, sizeof got));
    check("cudaLaunchKernelEx with 2-block clusters", got == 256);
  }

  int blocks = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, scaled, 256, 0));
  check("occupancy", blocks > 0);

  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(s));
  CK(cudaFree(da));
  CK(cudaFree(db));
  printf("runtime API: %s (%d failed)\n", failures ? "FAIL" : "ok", failures);
  return failures != 0;
}
