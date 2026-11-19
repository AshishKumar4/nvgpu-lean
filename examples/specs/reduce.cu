// Each CTA sums its 256 inputs: a shared-memory tree, one partial per CTA.
extern "C" __global__ void block_sum(const int* in, int* out) {
    __shared__ int s[256];
    int t = threadIdx.x;
    s[t] = in[blockIdx.x * 256 + t];
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) {
        if (t < k) s[t] += s[t + k];
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = s[0];
}

// The whole sum in one launch: each CTA writes its partial, then takes a ticket from a global
// counter; the CTA with the last ticket has every other partial and adds them (the CUDA samples'
// threadFenceReduction).
extern "C" __global__ void grid_sum(const int* in, int* partial, unsigned* counter, int* total) {
    __shared__ int s[256];
    __shared__ bool last;
    int t = threadIdx.x;
    s[t] = in[blockIdx.x * 256 + t];
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1) {
        if (t < k) s[t] += s[t + k];
        __syncthreads();
    }
    if (t == 0) {
        partial[blockIdx.x] = s[0];
        __threadfence();
        last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (last && t == 0) {
        int sum = 0;
        for (int i = 0; i < gridDim.x; ++i) sum += ((volatile int*)partial)[i];
        *total = sum;
    }
}
