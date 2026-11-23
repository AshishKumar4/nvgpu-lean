// Managed memory (cuMemAllocManaged): device memory the host also maps at the same address, so a
// pointer is good on both sides. The host's mapping holds the bytes between device work; a
// launch first takes the pages the host changed since the last (`managedPush`, against a shadow
// of what the device last saw), and after it gives the host the pages it changed
// (`managedPull`), so host writes to other pages meanwhile survive. Copies, sets and the emulated
// NCCL reach managed memory through the host's mapping. The host must not touch a page a running
// kernel writes (the device reports no concurrent managed access). Included by cuda.c.

#define MANAGED_PAGE 4096ull

typedef struct Managed {
  CUdeviceptr base;
  size_t size;      // mapped bytes, whole pages
  uint8_t *shadow;  // the bytes the device last saw
  int device;
  struct Managed *next;
} Managed;
static Managed *managedList;

// The managed allocation holding [a, a + n) whole (the lock held).
static Managed *managedAt(CUdeviceptr a, size_t n) {
  for (Managed *m = managedList; m; m = m->next)
    if (a >= m->base && a + n <= m->base + m->size) return m;
  return NULL;
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *p, size_t bytes, unsigned int flags) {
  NEED_INIT();
  if (!p || bytes == 0 || (flags != CU_MEM_ATTACH_GLOBAL && flags != CU_MEM_ATTACH_HOST)) return CUDA_ERROR_INVALID_VALUE;
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
  *m = (Managed){.base = *p, .size = size, .shadow = shadow, .device = gcur, .next = managedList};
  managedList = m;
  pthread_mutex_unlock(&lock);
  return CUDA_SUCCESS;
}

// Freeing managed memory: its mapping gone with it (the lock not held).
static void managedFree(CUdeviceptr p) {
  pthread_mutex_lock(&lock);
  for (Managed **q = &managedList; *q; q = &(*q)->next)
    if ((*q)->base == p) {
      Managed *m = *q;
      *q = m->next;
      munmap((void *)m->base, m->size);
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
      const uint8_t *h = (const uint8_t *)m->base + o;
      if (!memcmp(h, m->shadow + o, MANAGED_PAGE)) continue;
      size_t e = o + MANAGED_PAGE;
      while (e < m->size && memcmp((const uint8_t *)m->base + e, m->shadow + e, MANAGED_PAGE)) e += MANAGED_PAGE;
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
        memcpy((uint8_t *)m->base + o, d + o, MANAGED_PAGE);
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
