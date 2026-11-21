// The special-function unit's binary32 forms over a sweep of every exponent and class: each
// input's 2^x, 1/x, 1/sqrt(x) and log2(x), with subnormals kept and flushed (`ex2`, `rcp`,
// `rsqrt`, `lg2` `.approx[.ftz].f32`), eight results an input.
extern "C" __global__ void special(const unsigned* x, unsigned* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __uint_as_float(x[i]), r[8];
    asm("ex2.approx.f32 %0, %1;" : "=f"(r[0]) : "f"(v));
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r[1]) : "f"(v));
    asm("rcp.approx.f32 %0, %1;" : "=f"(r[2]) : "f"(v));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r[3]) : "f"(v));
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(r[4]) : "f"(v));
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r[5]) : "f"(v));
    asm("lg2.approx.f32 %0, %1;" : "=f"(r[6]) : "f"(v));
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r[7]) : "f"(v));
    for (int k = 0; k < 8; ++k) y[8 * i + k] = __float_as_uint(r[k]);
}
