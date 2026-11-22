// Divergent control flow that compilers lower to convergence barriers with BREAK: loops that
// lanes leave early around warp-synchronous operations, a break out of a loop from inside a
// conditional block. Each warp-synchronous operation inside them reads back only the lane's own
// value or bit, so the result does not depend on which lanes happen to run together; the final
// shuffle needs the whole warp back together.
extern "C" __global__ void control(const int* in, int* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned lane = threadIdx.x & 31u;
  int acc = 0;
  for (int i = 0; i < 64; ++i) {
    const int v = in[(t * 7 + i * 3) & 4095];
    if (v % 13 == 0) break;
    acc += __shfl_sync(__activemask(), v, lane);
    if (v % 5 == 0) {
      for (int j = 0; j < 8; ++j) {
        const int w = in[(v + j) & 4095];
        if (w % 3 == 0) break;
        acc ^= ((__ballot_sync(__activemask(), w & 1) >> lane) & 1u) << (j + i % 7);
      }
    }
  }
  for (int i = 0; i < 32; ++i) {
    const int w = in[(t * 5 + i * 11) & 4095];
    if (w & 1) {
      if ((w & 30) == 6) break;
      acc += __shfl_sync(__activemask(), w, lane) >> 3;
    }
    acc ^= (int)((__ballot_sync(__activemask(), (w >> 3) & 1) >> lane) & 1u) << (i % 9);
  }
  acc += __shfl_xor_sync(0xffffffffu, acc, 1);
  out[t] = acc;
}
