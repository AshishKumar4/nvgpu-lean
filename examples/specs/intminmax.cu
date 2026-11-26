// Integer minimum and maximum, packed (PTX min/max .u16x2, .s16x2, SASS VIMNMX.U16x2, .S16x2) and
// clamped at zero (.relu, VIMNMX.S16x2.RELU and VIMNMX.RELU on sm_100a), and saturating sums
// (add.sat.s32, sub.sat.s32), over pairs of words: ten words a pair.
extern "C" __global__ void intminmax(const unsigned* x, unsigned* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned a = x[2 * i], b = x[2 * i + 1], r[10];
    asm("min.u16x2 %0, %1, %2;" : "=r"(r[0]) : "r"(a), "r"(b));
    asm("max.u16x2 %0, %1, %2;" : "=r"(r[1]) : "r"(a), "r"(b));
    asm("min.s16x2 %0, %1, %2;" : "=r"(r[2]) : "r"(a), "r"(b));
    asm("max.s16x2 %0, %1, %2;" : "=r"(r[3]) : "r"(a), "r"(b));
    asm("min.relu.s16x2 %0, %1, %2;" : "=r"(r[4]) : "r"(a), "r"(b));
    asm("max.relu.s16x2 %0, %1, %2;" : "=r"(r[5]) : "r"(a), "r"(b));
    asm("min.relu.s32 %0, %1, %2;" : "=r"(r[6]) : "r"(a), "r"(b));
    asm("max.relu.s32 %0, %1, %2;" : "=r"(r[7]) : "r"(a), "r"(b));
    asm("add.sat.s32 %0, %1, %2;" : "=r"(r[8]) : "r"(a), "r"(b));
    asm("sub.sat.s32 %0, %1, %2;" : "=r"(r[9]) : "r"(a), "r"(b));
    for (int k = 0; k < 10; ++k) y[10 * i + k] = r[k];
}
