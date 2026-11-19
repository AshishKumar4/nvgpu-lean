// Warp collectives: shuffles, votes, match, redux and the active mask.
extern "C" __global__ void warpops(const unsigned* a, unsigned* out) {
  unsigned t = threadIdx.x, x = a[t];
  unsigned* o = out + 12 * t;
  o[0] = __shfl_sync(0xffffffffu, x, (t * 7) & 31);
  o[1] = __shfl_up_sync(0xffffffffu, x, 3);
  o[2] = __shfl_down_sync(0xffffffffu, x, 5, 16);
  o[3] = __shfl_xor_sync(0xffffffffu, x, 9);
  o[4] = __ballot_sync(0xffffffffu, x & 1);
  o[5] = __all_sync(0xffffffffu, x != 0);
  o[6] = __any_sync(0xffffffffu, x == 7);
  o[7] = __match_any_sync(0xffffffffu, x & 3);
  int pred;
  o[8] = __match_all_sync(0xffffffffu, x >> 30, &pred) | (pred << 31);
  o[9] = __reduce_add_sync(0xffffffffu, x);
  o[10] = __reduce_max_sync(0xffffffffu, x);
  o[11] = (x & 1) ? __activemask() : 0;
}
