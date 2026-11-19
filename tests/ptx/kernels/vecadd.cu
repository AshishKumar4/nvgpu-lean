// Element-wise kernels: one thread per element, bounds-checked.
extern "C" __global__ void vecadd_f32(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void saxpy(float alpha, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaf(alpha, x[i], y[i]);
}

extern "C" __global__ void vecadd_i64(const long long* a, const long long* b, long long* c, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    c[i] = a[i] + b[i];
}
