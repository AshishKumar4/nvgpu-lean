// Warpgroup MMAs of the kinds other than f16's (H100, sm_90a), from shared memory without
// swizzling (K-major), m64n64, one warpgroup: TF32 (k8), E4M3 A by E5M2 B (k32) into f32, and s8 A
// by u8 B (k32) into s32; and BF16 (k16) with A from registers (the RS form, its fragment as
// `mma.sync`'s), negated (imm-scale-a -1). A and B are small integers, so every sum is exact; out
// holds each D, 64 x 64 row-major, one after another.
#include <cuda_bf16.h>
constexpr int N = 64;

// Element (mn, k) of a K-major operand without swizzling, `bytes` an element: 8-row by 16-byte
// core matrices, the two along K 128 bytes apart (LBO), row groups 256 bytes apart (SBO).
__device__ int at(int mn, int k, int bytes) {
  const int per = 16 / bytes;
  return (mn / 8) * 256 + (k / per) * 128 + (mn % 8) * 16 + (k % per) * bytes;
}

__device__ unsigned long long gdesc(const void* p) {
  const unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  return (unsigned long long)((addr >> 4) & 0x3FFF) | (unsigned long long)(128 >> 4) << 16 |
         (unsigned long long)(256 >> 4) << 32;
}

// -3..3 as E4M3 and as E5M2.
__device__ unsigned char e4m3(int v) {
  const unsigned char m[4] = {0x00, 0x38, 0x40, 0x44};
  return (v < 0 ? 0x80 : 0) | m[v < 0 ? -v : v];
}
__device__ unsigned char e5m2(int v) {
  const unsigned char m[4] = {0x00, 0x3C, 0x40, 0x42};
  return (v < 0 ? 0x80 : 0) | m[v < 0 ? -v : v];
}

// This thread's 32 accumulator values (`accPos`) to D, 64 x 64 row-major at `out`.
__device__ void store(unsigned* out, const unsigned* d, int t) {
  for (int v = 0; v < 32; ++v)
    out[(16 * (t / 32) + (t % 32) / 4 + 8 * ((v / 2) % 2)) * N + 2 * (t % 4) + v % 2 + 8 * (v / 4)] = d[v];
}

extern "C" __global__ void wgmma_kinds(unsigned* out) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __shared__ alignas(128) unsigned char bh[N * 32];
  __shared__ alignas(128) unsigned char at32[64 * 32], bt32[N * 32], a8[64 * 32], b8[N * 32], ai[64 * 32], bi[N * 32];
  const int t = threadIdx.x;
  for (int e = t; e < 64 * 32; e += blockDim.x) {
    const int i = e / 32, k = e % 32, v = (i + 2 * k) % 5 - 2;
    if (k < 8) *(float*)(at32 + at(i, k, 4)) = (float)v;
    a8[at(i, k, 1)] = e4m3(v);
    ai[at(i, k, 1)] = (unsigned char)(signed char)v;
  }
  for (int e = t; e < N * 32; e += blockDim.x) {
    const int j = e / 32, k = e % 32, v = (3 * k + j) % 7 - 3;
    if (k < 8) *(float*)(bt32 + at(j, k, 4)) = (float)v;
    b8[at(j, k, 1)] = e5m2(v);
    bi[at(j, k, 1)] = (unsigned char)(v + 3);
    if (k < 16) *(__nv_bfloat16*)(bh + at(j, k, 2)) = __int2bfloat16_rn(v);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  unsigned d[32];
  unsigned long long da, db;
  for (int v = 0; v < 32; ++v) d[v] = 0;
  da = gdesc(at32); db = gdesc(bt32);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile("{ .reg .pred p; setp.ne.u32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
               : "l"(da), "l"(db), "r"(0) : "memory");
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  store(out, d, t);
  for (int v = 0; v < 32; ++v) d[v] = 0;
  da = gdesc(a8); db = gdesc(b8);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile("{ .reg .pred p; setp.ne.u32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e5m2 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1; }"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
               : "l"(da), "l"(db), "r"(0) : "memory");
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  store(out + 64 * N, d, t);
  for (int v = 0; v < 32; ++v) d[v] = 0;
  da = gdesc(ai); db = gdesc(bi);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile("{ .reg .pred p; setp.ne.u32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.u8 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p; }"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
               : "l"(da), "l"(db), "r"(0) : "memory");
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  store(out + 2 * 64 * N, d, t);
  for (int v = 0; v < 32; ++v) d[v] = 0;
  // This thread's A fragment: register r rows 16 (t / 32) + (t % 32) / 4 + 8 (r % 2), columns
  // 2 (t % 4) + 8 (r / 2) and the next, low first.
  unsigned ar[4];
  for (int r = 0; r < 4; ++r) {
    const int i = 16 * (t / 32) + (t % 32) / 4 + 8 * (r % 2), k = 2 * (t % 4) + 8 * (r / 2);
    ar[r] = (unsigned)__bfloat16_as_ushort(__int2bfloat16_rn((i + 2 * k) % 5 - 2)) |
            (unsigned)__bfloat16_as_ushort(__int2bfloat16_rn((i + 2 * k + 2) % 5 - 2)) << 16;
  }
  db = gdesc(bh);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile("{ .reg .pred p; setp.ne.u32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32, %33, %34, %35}, %36, p, -1, 1, 0; }"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
               : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "l"(db), "r"(0) : "memory");
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  store(out + 3 * 64 * N, d, t);
#endif
}
