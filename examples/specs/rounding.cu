// Rounding to an integral value (PTX `cvt.rXi`, SASS FRND) over a sweep of binary32 inputs of every
// exponent and class: to nearest even, toward zero, down and up, with subnormals kept and flushed,
// and the same four of the input widened to binary64 (F2F.F64.F32), and that narrowed back toward
// zero (F2F.F32.F64.RZ); the widened input times a binary64 parameter less it (a constant-bank
// pair and its negation as binary64 sources); the input narrowed to bfloat16; and converted to
// 16-bit integers, saturating (F2I.S16, F2I.U16); and the input rounded to binary16 and widened
// back (HADD2.F32); twenty-one words an input.
extern "C" __global__ void rounding(const unsigned* x, unsigned* y, int n, double s) {
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
    float back;
    asm("cvt.rz.f32.f64 %0, %1;" : "=f"(back) : "d"(w));
    for (int k = 0; k < 5; ++k) y[21 * i + k] = __float_as_uint(r[k]);
    y[21 * i + 5] = __float_as_uint(back);
    for (int k = 0; k < 4; ++k) {
        unsigned long long b = __double_as_longlong(d[k]);
        y[21 * i + 6 + 2 * k] = (unsigned)b;
        y[21 * i + 7 + 2 * k] = (unsigned)(b >> 32);
    }
    unsigned long long q = __double_as_longlong((double)v * s - s);
    y[21 * i + 14] = (unsigned)q;
    y[21 * i + 15] = (unsigned)(q >> 32);
    unsigned short b16, s0, s1, u0;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b16) : "f"(v));
    asm("cvt.rzi.s16.f32 %0, %1;" : "=h"(s0) : "f"(v));
    asm("cvt.rni.s16.f32 %0, %1;" : "=h"(s1) : "f"(v));
    asm("cvt.rzi.u16.f32 %0, %1;" : "=h"(u0) : "f"(v));
    y[21 * i + 16] = b16;
    y[21 * i + 17] = (unsigned)(int)(short)s0;
    y[21 * i + 18] = (unsigned)(int)(short)s1;
    y[21 * i + 19] = u0;
    unsigned short h16;
    float wide16;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h16) : "f"(v));
    asm("cvt.f32.f16 %0, %1;" : "=f"(wide16) : "h"(h16));
    y[21 * i + 20] = __float_as_uint(wide16);
}
