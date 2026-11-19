// y[i] = a * x[i] + y[i], one thread per element.
extern "C" __global__ void axpy(const int* x, int* y, int a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
