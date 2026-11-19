/* Prints PTX Math's outputs (docs/sources.md `ptxmath`) for the inputs `num-tests mufu` uses:
   one line `function input output` per case, inputs and outputs as binary32 encodings in hex. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "models.h"

static float fl(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }
static uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }

int main(void)
{
    const char *names[] = {"rcp", "sqrt", "rsqrt", "lg2", "ex2", "sin", "cos"};
    float (*fns[])(float) = {ptxm_rcp_sm5x, ptxm_sqrt_sm6x, ptxm_rsqrt_sm5x, ptxm_lg2_sm5x,
                             ptxm_ex2_sm5x, ptxm_sin_sm70, ptxm_cos_sm70};
    for (int k = 0; k < 7; k++)
        for (uint32_t i = 0; i < 65536; i++) {
            uint32_t xs[3] = {i * 65537u, 0x3f800000u + i, 0x3f000000u + i * 128u};
            for (int j = 0; j < 3; j++)
                printf("%s %x %x\n", names[k], xs[j], bits(fns[k](fl(xs[j]))));
        }
    return 0;
}
