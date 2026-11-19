// Floating-point instructions and conversions, from operand arrays a, b, c.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void floatops(const float* a, const float* b, const float* c,
                                   unsigned* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i], y = b[i], z = c[i];
  unsigned* o = out + 24 * i;
  o[0] = __float_as_uint(fmaf(x, y, z));
  o[1] = __float_as_uint(x / y);
  o[2] = __float_as_uint(sqrtf(x));
  o[3] = __float_as_uint(__frcp_rn(x));
  o[4] = __float_as_uint(fminf(x, y));
  o[5] = __float_as_uint(fmaxf(x, y));
  o[6] = (unsigned)__float2int_rn(x);
  o[7] = (unsigned)__float2int_rz(x);
  o[8] = __float2uint_rd(x);
  o[9] = __float_as_uint(__int2float_rn((int)__float_as_uint(y)));
  o[10] = __half_as_ushort(__float2half_rn(x));
  o[11] = __float_as_uint(__half2float(__ushort_as_half((unsigned short)__float_as_uint(y))));
  o[12] = __bfloat16_as_ushort(__float2bfloat16_rn(x));
  o[13] = __float_as_uint(__fadd_rz(x, y));
  o[14] = __float_as_uint(__fmul_ru(x, y));
  o[15] = __float_as_uint(__fdiv_rd(x, y));
  o[16] = __float_as_uint(rintf(x));
  o[17] = __float_as_uint(floorf(x));
  o[18] = __float_as_uint(fabsf(x) + -y);
  o[19] = (unsigned)(x < y) | ((unsigned)(x != x) << 1) | ((unsigned)isinf(y) << 2);
  o[20] = __float_as_uint(copysignf(x, y));
  o[21] = __float_as_uint(__saturatef(x));
  double d = (double)x * (double)y + (double)z;
  o[22] = (unsigned)__double2hiint(d);
  o[23] = (unsigned)__double2loint(d);
}
