// Managed memory (cuMemAllocManaged): device memory the host also maps at the same address, so a
// pointer is good on both sides. The host's mapping holds the bytes between device work; a
// launch first takes the pages the host changed since the last (`managedPush`, against a shadow
// of what the device last saw), and after it gives the host the pages it changed
// (`managedPull`), so host writes to other pages meanwhile survive. Copies, sets and the emulated
// NCCL reach managed memory through the host's mapping. The host must not touch a page a running
// kernel writes (the device reports no concurrent managed access). Mapped page-locked memory
// (cuMemHostAlloc with CU_MEMHOSTALLOC_DEVICEMAP) is the same, reported as host memory; so is
// registered host memory (cuMemHostRegister), whose bytes stay at the host's address while the
// device sees them at another of its own (`at`), only the registered bytes going back. Included by
// cuda.c.

#define MANAGED_PAGE 4096ull

typedef struct Managed {
  CUdeviceptr base;
  size_t size;      // mapped bytes, whole pages
  uint8_t *shadow;  // the bytes the device last saw
  int device;
  int host;         // page-locked host memory the device maps (cuMemHostAlloc), not managed
  uint8_t *at;      // where the host holds the bytes: `base`, or registered memory's own pages
  size_t lo, hi;    // of those, the bytes the host lets the device change (registered memory's)
  struct Managed *next;
} Managed;
static Managed *managedList;

// The managed allocation holding [a, a + n) whole (the lock held).
static Managed *managedAt(CUdeviceptr a, size_t n) {
  for (Managed *m = managedList; m; m = m->next)
    if (a >= m->base && a + n <= m->base + m->size) return m;
  return NULL;
}

// Memory both sides map at one address, `bytes` of it at *p (`host`: page-locked host memory).
static CUresult mapShared(CUdeviceptr *p, size_t bytes, int host) {
  size_t size = (bytes + MANAGED_PAGE - 1) / MANAGED_PAGE * MANAGED_PAGE;
  CUresult r = allocRange(p, size, ALLOC_ALIGN, 0);
  if (r != CUDA_SUCCESS) return r;
  void *h = mmap((void *)*p, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  Managed *m = h == (void *)*p ? calloc(1, sizeof *m) : NULL;
  uint8_t *shadow = m ? calloc(1, size) : NULL;
  if (!shadow) {
    if (h != MAP_FAILED) munmap(h, size);
    free(m);
    freeRange(*p, 0);
    note("the host cannot map managed memory at %#llx", (unsigned long long)*p);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  // Both sides start zero: the device's memory is set, the mapping and shadow are fresh.
  devSet(*p, 0, size);
  pthread_mutex_lock(&lock);
  *m = (Managed){.base = *p, .size = size, .shadow = shadow, .device = gcur, .host = host,
                 .at = (uint8_t *)*p, .lo = 0, .hi = size, .next = managedList};
  managedList = m;
  pthread_mutex_unlock(&lock);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *p, size_t bytes, unsigned int flags) {
  NEED_INIT();
  if (!p || bytes == 0 || (flags != CU_MEM_ATTACH_GLOBAL && flags != CU_MEM_ATTACH_HOST)) return CUDA_ERROR_INVALID_VALUE;
  return mapShared(p, bytes, 0);
}

// Registered host memory [p, p + bytes): device memory over its whole pages at *d, holding what
// the host's pages hold, which launches keep in step (`managedPush`, `managedPull`).
static CUresult mapRegistered(CUdeviceptr *d, void *p, size_t bytes) {
  uint8_t *first = (uint8_t *)((uintptr_t)p & ~(uintptr_t)(MANAGED_PAGE - 1));
  size_t size = ((uint8_t *)p + bytes - first + MANAGED_PAGE - 1) / MANAGED_PAGE * MANAGED_PAGE;
  CUresult r = allocRange(d, size, ALLOC_ALIGN, 0);
  if (r != CUDA_SUCCESS) return r;
  Managed *m = calloc(1, sizeof *m);
  uint8_t *shadow = m ? calloc(1, size) : NULL;
  if (!shadow) {
    free(m);
    freeRange(*d, 0);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  // The device starts as the shadow, zero; the first launch takes every page that differs.
  devSet(*d, 0, size);
  pthread_mutex_lock(&lock);
  *m = (Managed){.base = *d, .size = size, .shadow = shadow, .device = gcur, .host = 1, .at = first,
                 .lo = (uint8_t *)p - first, .hi = (uint8_t *)p - first + bytes, .next = managedList};
  managedList = m;
  pthread_mutex_unlock(&lock);
  *d += (uint8_t *)p - first;
  return CUDA_SUCCESS;
}

// The registered memory whose host bytes hold `p` (the lock held).
static Managed *registeredAt(const void *p) {
  for (Managed *m = managedList; m; m = m->next)
    if (m->at != (uint8_t *)m->base && (const uint8_t *)p >= m->at + m->lo && (const uint8_t *)p < m->at + m->hi)
      return m;
  return NULL;
}

// Freeing managed memory: its mapping gone with it (the lock not held).
static void managedFree(CUdeviceptr p) {
  pthread_mutex_lock(&lock);
  for (Managed **q = &managedList; *q; q = &(*q)->next)
    if ((*q)->base == p) {
      Managed *m = *q;
      *q = m->next;
      if (m->at == (uint8_t *)m->base) munmap((void *)m->base, m->size);
      free(m->shadow);
      free(m);
      break;
    }
  pthread_mutex_unlock(&lock);
}

// Before a launch: the pages the host changed since the device last saw them, to the device (the
// lock held).
static int managedPush(void) {
  for (Managed *m = managedList; m; m = m->next) {
    int was = gpuEnter(m->device);
    enterLean();
    for (size_t o = 0; o < m->size; o += MANAGED_PAGE) {
      const uint8_t *h = m->at + o;
      if (!memcmp(h, m->shadow + o, MANAGED_PAGE)) continue;
      size_t e = o + MANAGED_PAGE;
      while (e < m->size && memcmp(m->at + e, m->shadow + e, MANAGED_PAGE)) e += MANAGED_PAGE;
      size_t n = e - o;
      memcpy(m->shadow + o, h, n);
      lean_object *b = lean_alloc_sarray(1, n, n);
      memcpy(lean_sarray_cptr(b), h, n);
      if (!resWrote(m->base + o, n, 1) || !take(nvgpu_write(G.lean, m->base + o, b)) || !resWrote(m->base + o, n, 0)) {
        gpuLeave(was);
        return 0;
      }
      o = e - MANAGED_PAGE;
    }
    gpuLeave(was);
  }
  return 1;
}

// After a launch: the pages the device changed, to the host (the lock held).
static int managedPull(void) {
  for (Managed *m = managedList; m; m = m->next) {
    int was = gpuEnter(m->device);
    enterLean();
    lean_object *b = resPull(m->base, m->size) ? ioValue(nvgpu_read(lend(), m->base, m->size, lean_io_mk_world())) : NULL;
    gpuLeave(was);
    if (!b) return 0;
    const uint8_t *d = lean_sarray_cptr(b);
    for (size_t o = 0; o < m->size; o += MANAGED_PAGE)
      if (memcmp(d + o, m->shadow + o, MANAGED_PAGE)) {
        memcpy(m->shadow + o, d + o, MANAGED_PAGE);
        size_t a = o > m->lo ? o : m->lo, b = o + MANAGED_PAGE < m->hi ? o + MANAGED_PAGE : m->hi;
        if (a < b) memcpy(m->at + a, d + a, b - a);
      }
    lean_dec(b);
  }
  return 1;
}

// Prefetches and advice: no effect, the bytes being where the next work needs them either way.
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr p, size_t n, CUmemLocation where, unsigned int flags, CUstream s) {
  NEED_INIT();
  (void)p; (void)n; (void)where; (void)flags; (void)s;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemAdvise(CUdeviceptr p, size_t n, CUmem_advise a, CUmemLocation where) {
  NEED_INIT();
  (void)p; (void)n; (void)a; (void)where;
  return CUDA_SUCCESS;
}
