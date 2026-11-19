// Tiled transpose through shared memory, padded against bank conflicts.
#define TILE 32
#define ROWS 8
extern "C" __global__ void transpose_f32(float* out, const float* in, int width, int height) {
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    if (x < width && y + j < height) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    if (x < height && y + j < width) out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
}
