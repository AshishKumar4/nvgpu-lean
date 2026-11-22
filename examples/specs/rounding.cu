// Rounding to an integral value (PTX `cvt.rXi`, SASS FRND) over a sweep of binary32 inputs of every
// exponent and class: to nearest even, toward zero, down and up, with subnormals kept and flushed,
// and the same four of the input widened to binary64; twelve words an input.
extern "C" __global__ void rounding(const unsigned* x, unsigned* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __uint_as_float(x[i]), r[5];
    asm("cvt.rni.f32.f32 %0, %1;" : "=f"(r[0]) : "f"(v));
    asm("cvt.rzi.f32.f32 %0, %1;" : "=f"(r[1]) : "f"(v));
    asm("cvt.rmi.f32.f32 %0, %1;" : "=f"(r[2]) : "f"(v));
    asm("cvt.rpi.f32.f32 %0, %1;" : "=f"(r[3]) : "f"(v));
    asm("cvt.rni.ftz.f32.f32 %0, %1;" : "=f"(r[4]) : "f"(v));
    double w = (double)v * 1048576.0 + 0.25, d[4];
    asm("cvt.rni.f64.f64 %0, %1;" : "=d"(d[0]) : "d"(w));
    asm("cvt.rzi.f64.f64 %0, %1;" : "=d"(d[1]) : "d"(w));
    asm("cvt.rmi.f64.f64 %0, %1;" : "=d"(d[2]) : "d"(w));
    asm("cvt.rpi.f64.f64 %0, %1;" : "=d"(d[3]) : "d"(w));
    for (int k = 0; k < 5; ++k) y[16 * i + k] = __float_as_uint(r[k]);
    for (int k = 0; k < 4; ++k) {
        unsigned long long b = __double_as_longlong(d[k]);
        y[16 * i + 6 + 2 * k] = (unsigned)b;
        y[16 * i + 7 + 2 * k] = (unsigned)(b >> 32);
    }
}
