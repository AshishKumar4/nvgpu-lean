// Warp-level tensor-core MMA as XLA's and PyTorch's GEMMs issue it: ldmatrix of A (.x4) and of B
// transposed (.x2.trans) from shared memory; mma.sync m16n8k16 of f16 and of bf16 and m16n8k8 of
// tf32, into f32; a saturated epilogue (FFMA.SAT); and, sm_90 on, stmatrix of the f16 product's
// halves (.x2, and .x1.trans). Each warp works on its own 512 input words, pseudo-random bits. And
// a NaN-propagating maximum (FMNMX.NAN), byte and half parameters (narrow constant loads), 64-bit
// integers to floats (I2F.U64.RP, .S64, .F64.S64), and integer minima and maxima (IMNMX on sm_80).
extern "C" __global__ void mma_sync(const unsigned* in, unsigned* out, unsigned char s, unsigned short h) {
  __shared__ alignas(16) unsigned short a[4][16][16], b[4][16][8], st[4][16][8], st2[4][8][8];
  const int w = threadIdx.x / 32, l = threadIdx.x % 32, g = blockIdx.x * 4 + w;
  const unsigned* src = in + g * 512;
  unsigned* dst = out + g * 1024;
  for (int i = l; i < 256; i += 32) a[w][i / 16][i % 16] = (unsigned short)(src[i / 2] >> (16 * (i % 2)));
  for (int i = l; i < 128; i += 32) b[w][i / 8][i % 8] = (unsigned short)(src[128 + i / 2] >> (16 * (i % 2)));
  __syncwarp();
  unsigned ra[4], rb[2];
  const unsigned pa = (unsigned)__cvta_generic_to_shared(&a[w][l % 16][(l / 16) * 8]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3]) : "r"(pa));
  const unsigned pb = (unsigned)__cvta_generic_to_shared(&b[w][l % 16][0]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(rb[0]), "=r"(rb[1]) : "r"(pb));
  float c[4], d[4], e[4], f[4];
  for (int i = 0; i < 4; ++i) c[i] = __uint_as_float(src[192 + 4 * l + i] & 0xbfffffffu);
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
               : "=f"(e[0]), "=f"(e[1]), "=f"(e[2]), "=f"(e[3])
               : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  const unsigned t0 = src[320 + 4 * l], t1 = src[321 + 4 * l], t2 = src[322 + 4 * l], t3 = src[323 + 4 * l];
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
               : "=f"(f[0]), "=f"(f[1]), "=f"(f[2]), "=f"(f[3])
               : "r"(t0), "r"(t1), "r"(t2), "r"(t3), "r"(src[448 + 2 * l]), "r"(src[449 + 2 * l]),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  for (int i = 0; i < 4; ++i) {
    dst[4 * l + i] = __float_as_uint(d[i]);
    dst[128 + 4 * l + i] = __float_as_uint(e[i]);
    dst[256 + 4 * l + i] = __float_as_uint(f[i]);
    dst[384 + 4 * l + i] = __float_as_uint(__saturatef(fmaf(d[i], e[i], f[i])));
    float m;
    asm("max.NaN.f32 %0, %1, %2;" : "=f"(m) : "f"(d[i]), "f"(__uint_as_float(src[4 * l + i])));
    dst[640 + 4 * l + i] = __float_as_uint(m);
  }
  dst[768 + l] = (unsigned)s * (unsigned)l + (unsigned)h;
  const unsigned long long u = ((unsigned long long)src[2 * l] << 32) | src[2 * l + 1];
  dst[800 + l] = __float_as_uint(__ull2float_ru(u));
  dst[832 + l] = __float_as_uint(__ll2float_rz((long long)u));
  const double dd = __ll2double_rn((long long)u);
  dst[864 + 2 * l] = (unsigned)__double_as_longlong(dd);
  dst[865 + 2 * l] = (unsigned)(__double_as_longlong(dd) >> 32);
  dst[928 + l] = (unsigned)min((int)src[l], (int)src[l + 32]) ^ max(src[l + 64], src[l + 96]);
#if __CUDA_ARCH__ >= 900
  const unsigned h0 = (__float_as_uint(d[0]) >> 16) | (__float_as_uint(d[1]) & 0xffff0000u);
  const unsigned h1 = (__float_as_uint(d[2]) >> 16) | (__float_as_uint(d[3]) & 0xffff0000u);
  const unsigned ps = (unsigned)__cvta_generic_to_shared(&st[w][l % 16][0]);
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};" :: "r"(ps), "r"(h0), "r"(h1) : "memory");
  const unsigned pt = (unsigned)__cvta_generic_to_shared(&st2[w][l % 8][0]);
  asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};" :: "r"(pt), "r"(h0 ^ h1) : "memory");
  __syncwarp();
  for (int i = l; i < 64; i += 32) dst[512 + i] = ((const unsigned*)st[w])[i];
  for (int i = l; i < 32; i += 32) dst[576 + i] = ((const unsigned*)st2[w])[i];
#endif
}
