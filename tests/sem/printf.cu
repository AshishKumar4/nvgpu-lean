// Device printf: every lane prints its index; lane 3 also prints an int and a float (promoted to
// double, so 16 bytes of arguments with the padding).
#include <cstdio>

extern "C" __global__ void hello(int a, float b) {
  printf("t%d\n", threadIdx.x);
  if (threadIdx.x == 3) printf("a=%d b=%.3f\n", a, b);
}
