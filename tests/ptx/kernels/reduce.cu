// Sum reductions: warp shuffles, then shared memory across warps, then one atomic per block.
extern "C" __global__ void reduce_sum_i32(const int* in, int* out, int n) {
  __shared__ int partial[32];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int v = i < n ? in[i] : 0;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < (blockDim.x + 31) / 32 ? partial[lane] : 0;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) atomicAdd(out, v);
  }
}

extern "C" __global__ void reduce_max_f32(const float* in, float* out, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  float v = -__int_as_float(0x7f800000);
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x) v = fmaxf(v, in[i]);
  s[t] = v;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (t < k) s[t] = fmaxf(s[t], s[t + k]);
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = s[0];
}
