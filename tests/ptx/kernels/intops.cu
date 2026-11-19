// Integer and bit instructions, one result per op, from operand arrays a, b, c.
extern "C" __global__ void intops(const unsigned* a, const unsigned* b, const unsigned* c,
                                  unsigned* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned x = a[i], y = b[i], z = c[i];
  int sx = (int)x, sy = (int)y;
  unsigned* o = out + 24 * i;
  o[0] = __umulhi(x, y);
  o[1] = (unsigned)__mulhi(sx, sy);
  o[2] = y ? x / y : 0;
  o[3] = y ? x % y : 0;
  o[4] = (sy != 0 && !(sx == (int)0x80000000 && sy == -1)) ? (unsigned)(sx / sy) : 0;
  o[5] = (sy != 0 && !(sx == (int)0x80000000 && sy == -1)) ? (unsigned)(sx % sy) : 0;
  o[6] = __popc(x);
  o[7] = __clz(x);
  o[8] = __brev(x);
  o[9] = __byte_perm(x, y, z);
  o[10] = __funnelshift_l(x, y, z);
  o[11] = __funnelshift_rc(x, y, z);
  o[12] = (unsigned)max(sx, sy);
  o[13] = min(x, y);
  o[14] = (unsigned)abs(sx);
  o[15] = __sad(sx, sy, z);
  o[16] = (unsigned)__dp4a(sx, sy, (int)z);
  o[17] = (unsigned)__mul24(sx, sy);
  o[18] = x * y + z;
  o[19] = (x >> (y & 31)) | (x << ((32 - (y & 31)) & 31));
  o[20] = (unsigned)(sx >> (y & 31));
  o[21] = __ffs(sx);
  o[22] = (unsigned)(((unsigned long long)x * y) >> 7);
  o[23] = __vabsdiffu4(x, y);
}
