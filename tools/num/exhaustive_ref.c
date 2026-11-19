/* Reference for the exhaustive 16-bit arithmetic check (tools/num/exhaustive.sh): every pair of
   binary16 or bfloat16 operands, for add, sub, mul and div in each IEEE rounding mode, computed on
   the host in binary64 (exact for add, sub and mul; one rounding for div, which binary64's 53 bits
   make innocuous before rounding to 11 or 8) and rounded to the format here. NaN results are the
   canonical 0x7fff. Prints, per operation and mode, an FNV-1a hash of the results in order.

   Usage: exhaustive_ref FORMAT SHARD SHARDS   (FORMAT f16 or bf16; the first operand ranges over
   the SHARD-th of SHARDS equal parts of the encodings, the second over all of them). */
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON

static int EXP, MAN;

static double decode(uint32_t b)
{
    int bias = (1 << (EXP - 1)) - 1;
    uint32_t s = (b >> (EXP + MAN)) & 1, e = (b >> MAN) & ((1u << EXP) - 1), m = b & ((1u << MAN) - 1);
    double v;
    if (e == (1u << EXP) - 1) v = m ? NAN : INFINITY;
    else if (e == 0) v = ldexp((double)m, 1 - bias - MAN);
    else v = ldexp((double)(m | (1u << MAN)), (int)e - bias - MAN);
    return s ? -v : v;
}

enum { RN, RZ, RM, RP };

static uint32_t encode(double x, int mode)
{
    if (isnan(x)) return 0x7fff;
    int bias = (1 << (EXP - 1)) - 1, qmin = 1 - bias - MAN;
    uint32_t sign = signbit(x) ? 1u << (EXP + MAN) : 0;
    double a = fabs(x);
    uint32_t inf = sign | (((1u << EXP) - 1) << MAN), max = inf - 1;
    if (isinf(a)) return inf;
    if (a == 0) return sign;
    int E;
    frexp(a, &E);
    int q = E - 1 - MAN > qmin ? E - 1 - MAN : qmin;
    double n = ldexp(a, -q), t = floor(n), rem = n - t;
    int up = 0;
    switch (mode) {
    case RN: up = rem > 0.5 || (rem == 0.5 && fmod(t, 2) == 1); break;
    case RZ: up = 0; break;
    case RM: up = sign && rem > 0; break;
    case RP: up = !sign && rem > 0; break;
    }
    uint64_t m = (uint64_t)t + up;
    if (m == (1ull << (MAN + 1))) { m = 1ull << MAN; q++; }
    uint32_t ef = m < (1ull << MAN) ? 0 : (uint32_t)(q - qmin + 1);
    if (ef >= (1u << EXP) - 1) {
        int away = mode == RN || (mode == RM && sign) || (mode == RP && !sign);
        return away ? inf : max;
    }
    return sign | (ef << MAN) | (uint32_t)(m & ((1u << MAN) - 1));
}

int main(int argc, char **argv)
{
    if (argc != 4) return 2;
    if (!strcmp(argv[1], "f16")) { EXP = 5; MAN = 10; } else { EXP = 8; MAN = 7; }
    unsigned shard = atoi(argv[2]), shards = atoi(argv[3]), per = 65536 / shards;
    const char *ops[] = {"add", "sub", "mul", "div"}, *modes[] = {"rn", "rz", "rm", "rp"};
    const int fe[] = {FE_TONEAREST, FE_TOWARDZERO, FE_DOWNWARD, FE_UPWARD};
    for (int op = 0; op < 4; op++)
        for (int mode = 0; mode < 4; mode++) {
            fesetround(fe[mode]);
            uint64_t h = 0xcbf29ce484222325ull;
            for (uint32_t a = shard * per; a < (shard + 1) * per; a++)
                for (uint32_t b = 0; b < 65536; b++) {
                    volatile double x = decode(a), y = decode(b), r;
                    switch (op) {
                    case 0: r = x + y; break;
                    case 1: r = x - y; break;
                    case 2: r = x * y; break;
                    default: r = x / y; break;
                    }
                    h = (h ^ encode(r, mode)) * 0x100000001b3ull;
                }
            fesetround(FE_TONEAREST);
            printf("%s %s %s %016llx\n", argv[1], ops[op], modes[mode], (unsigned long long)h);
        }
    return 0;
}
