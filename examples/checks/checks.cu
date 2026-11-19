// Kernels for the simulator's own checks (`lake exe sim --races --unwritten`): each spec here
// launches one, and `ci/rows/checks.sh` states what the checks must report for it.

// Every thread adds its index into shared word 0 with no atomic and no barrier between: threads
// of the CTA write the same bytes in one barrier interval (a race).
extern "C" __global__ void shared_race(int* out) {
  __shared__ int s[1];
  if (threadIdx.x == 0) s[0] = 0;
  __syncthreads();
  s[0] = s[0] + (int)threadIdx.x;
  __syncthreads();
  if (threadIdx.x == 0) out[0] = s[0];
}

// The same sum, ordered: each thread writes its own word, a barrier, then thread 0 reads them all.
extern "C" __global__ void shared_ordered(int* out) {
  __shared__ int s[256];
  s[threadIdx.x] = (int)threadIdx.x;
  __syncthreads();
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < blockDim.x; ++i) sum += s[i];
    out[0] = sum;
  }
}

// Thread i writes global word i, then thread i reads word i + 1 with no barrier between (a race),
// and writes what it read to `out`.
extern "C" __global__ void global_race(int* buf, int* out) {
  const int i = threadIdx.x;
  buf[i] = i;
  out[i] = buf[(i + 1) % blockDim.x];
}

// Writes only the first half of `out`.
extern "C" __global__ void half_written(int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n / 2) out[i] = i;
}

// Two warpgroups of a 384-thread CTA; warpgroup 0 (the producer) does not take part. Consumer
// warpgroup 1 rewrites the first half of a shared slot, then both consumers meet at named barrier 1
// for 256 threads, then warpgroup 2 reads the first half and rewrites the second, which
// warpgroup 1 reads after a second meeting: ordered by the named barrier alone.
extern "C" __global__ void named_bar_ordered(int* out) {
  __shared__ int v[256];
  const int wg = threadIdx.x / 128, t = threadIdx.x % 128;
  if (wg == 0) return;
  if (wg == 1) v[t] = t;
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 2) { out[t] = v[t]; v[128 + t] = 2 * t; }
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 1) out[128 + t] = v[128 + t];
}

// The same without the second meeting: warpgroup 1 reads what warpgroup 2 writes, unordered.
extern "C" __global__ void named_bar_race(int* out) {
  __shared__ int v[256];
  const int wg = threadIdx.x / 128, t = threadIdx.x % 128;
  if (wg == 0) return;
  if (wg == 1) v[t] = t;
  asm volatile("bar.sync 1, 256;" ::: "memory");
  if (wg == 2) { out[t] = v[t]; v[128 + t] = 2 * t; }
  if (wg == 1) out[128 + t] = v[128 + t];
}

// Warp 0 writes a shared slot then arrives on an mbarrier; warp 1 waits on its phase, then reads
// the slot: ordered by the mbarrier (release arrive, acquire try_wait).
extern "C" __global__ void mbar_ordered(int* out) {
  __shared__ int v[32];
  __shared__ alignas(8) unsigned long long bar;
  const unsigned a = (unsigned)__cvta_generic_to_shared(&bar);
  if (threadIdx.x == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" :: "r"(a) : "memory");
  __syncthreads();
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  if (w == 0) {
    v[l] = l + 1;
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
  } else {
    unsigned done = 0;
    while (!done)
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(a) : "memory");
    out[l] = v[l];
  }
}

// The same, warp 1 reading before it waits: a race.
extern "C" __global__ void mbar_race(int* out) {
  __shared__ int v[32];
  __shared__ alignas(8) unsigned long long bar;
  const unsigned a = (unsigned)__cvta_generic_to_shared(&bar);
  if (threadIdx.x == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" :: "r"(a) : "memory");
  __syncthreads();
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  if (w == 0) {
    v[l] = l + 1;
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
  } else {
    out[l] = v[l];
    unsigned done = 0;
    while (!done)
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(a) : "memory");
  }
}

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
__device__ unsigned long long smem_desc(const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  return (unsigned long long)((a >> 4) & 0x3FFF) | (8ull << 16) | (16ull << 32);
}

// One warpgroup writes wgmma's operands to shared memory with ordinary stores, meets at a CTA
// barrier, then a wgmma reads them: with `fence.proxy.async` after the stores (FENCED) or without.
// REUSE: after `wgmma.wait_group 0` and a barrier, the last warp overwrites B, which the completed
// wgmma read: ordered by the wait and the barrier.
template <bool FENCED, bool REUSE = false>
__device__ void wgmma_operands(float* out) {
  __shared__ alignas(128) unsigned short a[64 * 16], b[8 * 16];
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) a[i] = 0x3F80;
  for (int i = threadIdx.x; i < 8 * 16; i += blockDim.x) b[i] = 0x3F80;
  if (FENCED) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  float d[4] = {0, 0, 0, 0};
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, %4, %5, 0, 1, 1, 0, 0;"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(smem_desc(a)), "l"(smem_desc(b)));
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  out[threadIdx.x] = d[0];
  if (REUSE) {
    __syncthreads();
    if (threadIdx.x >= 96) b[threadIdx.x - 96] = 0;
  }
}
#endif

extern "C" __global__ void wgmma_fenced(float* out) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  wgmma_operands<true>(out);
#endif
}

extern "C" __global__ void wgmma_reuse(float* out) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  wgmma_operands<true, true>(out);
#endif
}

extern "C" __global__ void wgmma_unfenced(float* out) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  wgmma_operands<false>(out);
#endif
}

// Thread 0 copies 128 words from global memory into a shared slot with one bulk copy whose
// completion signals an mbarrier (complete-tx); every thread waits on the mbarrier's phase, then
// reads the slot: the async proxy's writes ordered by the complete-tx.
template <bool WAIT>
__device__ void bulk_copy(const int* in, int* out) {
  __shared__ alignas(128) int v[128];
  __shared__ alignas(8) unsigned long long bar;
  const unsigned a = (unsigned)__cvta_generic_to_shared(&bar);
  const unsigned d = (unsigned)__cvta_generic_to_shared(v);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(a) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 512;" :: "r"(a) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 512, [%2];"
                 :: "r"(d), "l"(in), "r"(a) : "memory");
  }
  // Without WAIT, the slot is read before the wait: unordered with the copy's writes.
  const int x = WAIT ? 0 : v[threadIdx.x];
  unsigned done = 0;
  while (!done)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(a) : "memory");
  out[threadIdx.x] = WAIT ? v[threadIdx.x] : x;
}

extern "C" __global__ void bulk_ordered(const int* in, int* out) {
#if __CUDA_ARCH__ >= 900
  bulk_copy<true>(in, out);
#endif
}

extern "C" __global__ void bulk_race(const int* in, int* out) {
#if __CUDA_ARCH__ >= 900
  bulk_copy<false>(in, out);
#endif
}

// Each thread copies its word into a shared slot with `cp.async`, then the CTA meets at a barrier
// and thread 0 sums the slot: with `cp.async.wait_group 0` before the barrier (WAIT, ordered: the
// group's completion is the thread's, which the barrier hands on) or without.
template <bool WAIT>
__device__ void cp_async_sum(const int* in, int* out) {
  __shared__ alignas(16) int v[128];
  const unsigned d = (unsigned)__cvta_generic_to_shared(&v[threadIdx.x]);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d), "l"(in + threadIdx.x) : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
  if (WAIT) asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < blockDim.x; ++i) sum += v[i];
    out[0] = sum;
  }
  if (!WAIT) asm volatile("cp.async.wait_group 0;" ::: "memory");
}

extern "C" __global__ void cp_async_ordered(const int* in, int* out) { cp_async_sum<true>(in, out); }
extern "C" __global__ void cp_async_race(const int* in, int* out) { cp_async_sum<false>(in, out); }

// A cluster of two: each CTA's warp 0 writes words its warp 1 reads after the cluster barrier
// (`barrier.cluster.arrive.release` / `wait.acquire`), which orders the threads of each CTA.
extern "C" __global__ void __cluster_dims__(2, 1, 1) cluster_ordered(int* out) {
  __shared__ int v[32];
  const int warp = threadIdx.x / 32, t = threadIdx.x % 32;
  if (warp == 0) v[t] = t + 1;
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (warp == 1) out[blockIdx.x * 32 + t] = v[t];
}

// The same reads with no barrier between: a race on shared memory.
extern "C" __global__ void __cluster_dims__(2, 1, 1) cluster_race(int* out) {
  __shared__ int v[32];
  const int warp = threadIdx.x / 32, t = threadIdx.x % 32;
  if (warp == 0) v[t] = t + 1;
  if (warp == 1) out[blockIdx.x * 32 + t] = v[t];
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// A three-stage `cp.async` pipeline as CUTLASS's multistage mainloop runs one: two stages issued
// ahead, then per tile `cp.async.wait_group 1` (all but the newest group complete), a barrier,
// thread 0 summing the tile's stage (every thread's words), and the next stage issued into the
// slot the previous tile read. Ordered: each stage is complete before the barrier its readers pass,
// and refilled only after the barrier that follows its reads.
extern "C" __global__ void cp_async_stages(const int* in, int* out) {
  __shared__ alignas(16) int v[3][128];
  const int t = threadIdx.x, tiles = 6;
  auto issue = [&](int k) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(&v[k % 3][t]);
    if (k < tiles) asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d), "l"(in + k * 128 + t) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  issue(0);
  issue(1);
  int sum = 0;
  for (int k = 0; k < tiles; ++k) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncthreads();
    if (t == 0)
      for (int i = 0; i < 128; ++i) sum += v[k % 3][i];
    issue(k + 2);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  if (t == 0) out[0] = sum;
}

// Every CTA stores its own index into the same 32 words: stores of two CTAs to one global byte.
extern "C" __global__ void cross_cta_store(int* out) {
  if (threadIdx.x < 32) out[threadIdx.x] = blockIdx.x;
}

// Thread t reads the shared word thread t + 32 writes, which no thread writes in the last warp:
// a read of shared memory its CTA never wrote.
extern "C" __global__ void shared_unwritten(int* out) {
  __shared__ int v[96];
  if (threadIdx.x < 64) v[threadIdx.x] = threadIdx.x;
  __syncthreads();
  out[threadIdx.x] = v[threadIdx.x + 32 < 96 ? threadIdx.x + 32 : 95];
}

// Each CTA reads the word its predecessor stores: a global byte one CTA read and another stored.
extern "C" __global__ void cross_cta_read(int* data) {
  if (threadIdx.x < 32) {
    int v = data[blockIdx.x * 32 + threadIdx.x];
    data[(blockIdx.x + 1) * 32 + threadIdx.x] = v + 1;
  }
}

// Every CTA adds its index to one float: floating-point atomic additions from several warps.
extern "C" __global__ void float_atomic_sum(float* out) {
  if (threadIdx.x == 0) atomicAdd(out, 0.1f * blockIdx.x);
}
