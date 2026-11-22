// What reductions such as PyTorch's variance compile to: IEEE division (the fast path's check,
// FCHK, and the slow path called as a subroutine), square roots, float to and from integer
// conversions with directed rounding, and a Welford combination of per-lane (count, mean, M2)
// across the warp by shuffles; and the unbiased variance's divisor as PyTorch forms it, the count
// less a float parameter (FADD with a negated constant operand).
extern "C" __global__ void welford(const float* in, float* out, int n, float correction) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  float mean = 0.f, m2 = 0.f;
  unsigned count = 0;
  for (int i = t; i < n; i += gridDim.x * blockDim.x) {
    const float x = in[i];
    count += 1;
    const float d = x - mean;
    mean += d / (float)count;
    m2 += d * (x - mean);
  }
  for (int o = 16; o > 0; o /= 2) {
    const unsigned nb = __shfl_down_sync(0xffffffffu, count, o);
    const float mb = __shfl_down_sync(0xffffffffu, mean, o), m2b = __shfl_down_sync(0xffffffffu, m2, o);
    const unsigned nn = count + nb;
    const float d = mb - mean, f = nn ? (float)nb / (float)nn : 0.f;
    mean += d * f;
    m2 += m2b + d * d * (float)count * f;
    count = nn;
  }
  const float x = in[t % n];
  out[8 * t + 0] = mean;
  const float divisor = (float)count > correction ? (float)count - correction : 0.f;
  out[8 * t + 1] = m2 / divisor;
  out[8 * t + 2] = sqrtf(fabsf(m2));
  out[8 * t + 3] = rsqrtf(fabsf(x) + 1e-30f);
  out[8 * t + 4] = __uint_as_float(__float2uint_rz(fabsf(x) * 1e6f));
  out[8 * t + 5] = __uint2float_ru(__float_as_uint(x));
  out[8 * t + 6] = __ull2float_rd((unsigned long long)__float_as_uint(x) * 2654435761ull);
  out[8 * t + 7] = x / (x * 1e-38f + 3.0f);
}
