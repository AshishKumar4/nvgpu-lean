// Streams, events, copies, sets and launches of the emulated driver. Included by cuda.c.
//
// A submission owns its completion and retains the completions it depends on. Dependencies
// are captured under workLock, so later submissions cannot delay earlier legacy-stream work.
// Stream workers execute independently; callbacks receive the device's sticky error, while
// device work and ordinary host functions stop after failure.

typedef struct Completion {
  int refs, done;
} Completion;

static Completion *completionNew(void) {
  Completion *c = calloc(1, sizeof *c);
  c->refs = 1;
  return c;
}
static void completionRetain(Completion *c) {
  if (c) __atomic_add_fetch(&c->refs, 1, __ATOMIC_RELAXED);
}
static void completionRelease(Completion *c) {
  if (c && __atomic_sub_fetch(&c->refs, 1, __ATOMIC_ACQ_REL) == 0) free(c);
}
static void completionReplace(Completion **slot, Completion *c) {
  completionRetain(c);
  completionRelease(*slot);
  *slot = c;
}

typedef enum {
  OP_KERNEL, OP_COPY, OP_SET, OP_HOST, OP_CALLBACK, OP_RECORD, OP_WAIT, OP_FREE, OP_GRAPH,
  OP_WRITE_VALUE, OP_WAIT_VALUE, OP_NOP
} OpKind;

typedef struct Op {
  OpKind kind;
  struct Op *next;
  Completion *completion, **before;
  size_t nbefore;
  // A launch: the function, shapes and its parameter block.
  CUfunction f;
  unsigned grid[3], block[3], cluster[3], smem;
  unsigned char *params;
  uint32_t paramBytes;
  // A copy or set: rows of `width` bytes, `height` rows a slice, `depth` slices; each side's
  // address, row pitch and slice pitch. A set writes `value` of `elem` bytes.
  uintptr_t dst, src;
  size_t width, height, depth, dpitch, spitch, dslice, sslice;
  uint32_t value;
  unsigned elem;
  // A host function or stream callback.
  CUhostFn fn;
  CUstreamCallback cb;
  CUstream stream;
  void *user;
  // An event recording or wait.
  CUevent ev;
  uint64_t seq;
  // A graph launch.
  CUgraphExec exec;
  CUgraph graph;  // immutable snapshot owned by this launch
  // A stream memory operation: 32- or 64-bit value at `dst`, and the wait's condition.
  uint64_t value64;
  unsigned flags;
  int is64;
  int gpu;  // the device it runs on: its stream's (the submitting thread's on the legacy stream)
} Op;

struct CUstream_st {
  Op *head, *tail;
  Completion *last;
  int running;    // its thread is running `head`
  int started, dying;
  unsigned flags;
  int priority;
  unsigned long long id;
  pthread_t thread;
  // Stream capture (graph.h): the graph being captured into, the nodes the next one depends on.
  CUgraph capture;
  CUstream captureOrigin;
  int captureOwnsGraph;
  CUstreamCaptureMode captureMode;
  unsigned long long captureId;
  CUgraphNode *deps;
  size_t ndeps, depsCap;
  int captureBroken;
  struct CUstream_st *nextStream;
  int gpu;  // its context's device
};

struct CUevent_st {
  // Recordings enqueued, numbered from 1. Recordings on different streams finish in any order:
  // `done` is the number below which all have run, `ahead` the later ones that have.
  uint64_t recorded, done;
  uint64_t *ahead;
  size_t nahead, aheadCap;
  uint64_t cycles;          // the device's cycles when recording `cyclesOf` ran
  uint64_t cyclesOf;
  unsigned flags;
  int refs;                 // the handle and every pending recording or wait (atomic)
  // Recorded in a stream capture: the capture's graph and the nodes the recording followed.
  CUgraph capture;
  CUgraphNode *nodes;
  unsigned long long captureId;
  size_t nnodes;
};

// Guards the queues and events; `workCv` is signalled whenever work completes or arrives.
static pthread_mutex_t workLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t workCv = PTHREAD_COND_INITIALIZER;
static uint64_t epoch;  // pieces of work completed: a memory wait rechecks when it moves
static struct CUstream_st *streams;
static unsigned long long streamIds = 2;
static Completion *legacyLast;

static int isLegacy(CUstream s) {
  return s == NULL || s == CU_STREAM_LEGACY || s == CU_STREAM_PER_THREAD;
}

static CUresult captureOp(CUstream s, Op *o);  // graph.h
static CUresult runExec(Op *o);              // graph.h
static CUresult accelKernel(Op *o);          // accel.h
static CUresult accelChecked(Op *o);         // accel.h
static const char *accelArch(void);          // accel.h
static void graphFree(CUgraph g);
static void armLaunch(Op *o);                // workLock held
static CUresult captureAlloc(CUstream s, CUdeviceptr *p, size_t n);

// Event and executable references are atomic: work is copied and freed with and without
// `workLock` held.
static void eventRetain(CUevent e) { __atomic_add_fetch(&e->refs, 1, __ATOMIC_RELAXED); }
static void eventRelease(CUevent e) {
  if (__atomic_sub_fetch(&e->refs, 1, __ATOMIC_ACQ_REL) == 0) {
    free(e->ahead);
    free(e->nodes);
    free(e);
  }
}
static void execRelease(CUgraphExec e);  // graph.h
static void execRetain(CUgraphExec e);

// Recording `seq` of `e` has run (workLock held).
static int eventDone(CUevent e, uint64_t seq) {
  if (seq <= e->done) return 1;
  for (size_t i = 0; i < e->nahead; i++)
    if (e->ahead[i] == seq) return 1;
  return 0;
}
static void eventMark(CUevent e, uint64_t seq, uint64_t cycles) {
  if (eventDone(e, seq)) return;
  if (seq >= e->cyclesOf) {
    e->cycles = cycles;
    e->cyclesOf = seq;
  }
  if (e->nahead == e->aheadCap)
    e->ahead = realloc(e->ahead, sizeof *e->ahead * (e->aheadCap = e->aheadCap ? 2 * e->aheadCap : 4));
  e->ahead[e->nahead++] = seq;
  // Advance `done` over the recordings now contiguous.
  for (int moved = 1; moved;) {
    moved = 0;
    for (size_t i = 0; i < e->nahead; i++)
      if (e->ahead[i] == e->done + 1) {
        e->done++;
        e->ahead[i] = e->ahead[--e->nahead];
        moved = 1;
        break;
      }
  }
}

static void opFree(Op *o) {
  if (o->ev) eventRelease(o->ev);
  if (o->kind == OP_GRAPH && o->exec) execRelease(o->exec);
  if (o->graph) graphFree(o->graph);
  completionRelease(o->completion);
  for (size_t i = 0; i < o->nbefore; i++) completionRelease(o->before[i]);
  free(o->before);
  free(o->params);
  free(o);
}

// A copy of `o` (its parameter block too): graphs keep their nodes' work and run copies of it.
static Op *opCopy(const Op *o) {
  Op *c = malloc(sizeof *c);
  *c = *o;
  c->next = NULL;
  c->completion = NULL;
  c->before = NULL;
  c->nbefore = 0;
  if (o->params) {
    c->params = malloc(o->paramBytes ? o->paramBytes : 1);
    memcpy(c->params, o->params, o->paramBytes);
  }
  if (o->ev) eventRetain(o->ev);
  if (o->kind == OP_GRAPH && o->exec) execRetain(o->exec);
  return c;
}

static Op *newOp(OpKind k) {
  Op *o = calloc(1, sizeof *o);
  o->kind = k;
  o->depth = 1;
  o->height = 1;
  return o;
}

// ---- Running work ----

static CUresult runKernel(Op *o) {
  CUfunction f = o->f;
  // Made on this device when it first runs here (a library's kernel runs on any device).
  if (f->k[gcur] < 0) {
    CUresult r = resolve(f);
    if (r != CUDA_SUCCESS) return r;
  }
  pthread_mutex_lock(&lock);
  enterLean();
  if (state) state->busy = 1;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  uint64_t c0 = G.lean ? nvgpu_device_cycles(lend()) : 0;
  CUresult r = CUDA_SUCCESS;
  if (accelArch())
    r = getenv("NVGPU_LEAN_ACCEL_CHECK") ? accelChecked(o) : accelKernel(o);
  else if (ngpus > 1) {
    // A mesh: the kernel may reach a peer's memory, so it runs over every device's.
    lean_object *pb = lean_alloc_sarray(1, o->paramBytes, o->paramBytes);
    memcpy(lean_sarray_cptr(pb), o->params, o->paramBytes);
    lean_object *ds = lean_alloc_array(ngpus, ngpus);
    for (int d = 0; d < ngpus; d++) {
      lean_array_set_core(ds, d, gpus[d].lean);
      gpus[d].lean = NULL;
    }
    lean_object *res = ioValue(nvgpu_launch_mesh(ds, (uint32_t)gcur, (uint32_t)f->k[gcur], o->grid[0], o->grid[1],
                                                 o->grid[2], o->block[0], o->block[1], o->block[2],
                                                 o->cluster[0], o->cluster[1], o->cluster[2], o->smem, pb,
                                                 functional, lean_io_mk_world()));
    if (!takeMesh(res)) r = CUDA_ERROR_LAUNCH_FAILED;
  } else {
    lean_object *pb = lean_alloc_sarray(1, o->paramBytes, o->paramBytes);
    memcpy(lean_sarray_cptr(pb), o->params, o->paramBytes);
    lean_object *res = ioValue(nvgpu_launch(G.lean, (uint32_t)f->k[gcur], o->grid[0], o->grid[1], o->grid[2],
                                            o->block[0], o->block[1], o->block[2], o->cluster[0],
                                            o->cluster[1], o->cluster[2], o->smem, pb, functional,
                                            lean_io_mk_world()));
    if (!res) {
      G.lean = NULL;  // the device was the call's, and the call failed
      r = CUDA_ERROR_LAUNCH_FAILED;
    } else if (!take(res))
      r = CUDA_ERROR_LAUNCH_FAILED;
  }
  if (state) {
    state->busy = 0;
    state->launches++;
  }
  if (r != CUDA_SUCCESS)
    fprintf(stderr, "nvgpu-lean: launch of %s, grid %u %u %u, block %u %u %u, cluster %u %u %u, %u bytes of dynamic shared memory, failed: %s\n",
            f->name, o->grid[0], o->grid[1], o->grid[2], o->block[0], o->block[1], o->block[2], o->cluster[0],
            o->cluster[1], o->cluster[2], o->smem, lastError);
  else if (getenv("NVGPU_LEAN_VERBOSE")) {
    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint64_t c1 = G.lean ? nvgpu_device_cycles(lend()) : c0;
    fprintf(stderr, "nvgpu-lean: launch of %s, grid %u %u %u, block %u %u %u, cluster %u %u %u, %u bytes of dynamic shared memory: %.3f s, %llu cycles\n",
            f->name, o->grid[0], o->grid[1], o->grid[2], o->block[0], o->block[1], o->block[2], o->cluster[0],
            o->cluster[1], o->cluster[2], o->smem,
            (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec),
            (unsigned long long)(c1 - c0));
  }
  pthread_mutex_unlock(&lock);
  return r;
}

static int onDevice(uintptr_t p) { return isDevice((CUdeviceptr)p); }

// One contiguous run of bytes, by where each side is.
static CUresult copyBytes(uintptr_t dst, uintptr_t src, size_t n) {
  if (n == 0) return CUDA_SUCCESS;
  int dd = onDevice(dst), sd = onDevice(src);
  if (dd && sd) return devCopy(dst, src, n);
  if (dd) return devWrite(dst, (const void *)src, n);
  if (sd) return devRead((void *)dst, src, n);
  memmove((void *)dst, (const void *)src, n);
  return CUDA_SUCCESS;
}

static CUresult runCopy(Op *o) {
  if (o->dpitch == o->width && o->spitch == o->width && o->dslice == o->width * o->height &&
      o->sslice == o->width * o->height)
    return copyBytes(o->dst, o->src, o->width * o->height * o->depth);
  for (size_t z = 0; z < o->depth; z++)
    for (size_t y = 0; y < o->height; y++) {
      CUresult r = copyBytes(o->dst + z * o->dslice + y * o->dpitch,
                             o->src + z * o->sslice + y * o->spitch, o->width);
      if (r != CUDA_SUCCESS) return r;
    }
  return CUDA_SUCCESS;
}

// Bytes at `p`, device or host memory (page-locked host memory is addressable, as with unified
// addressing): written from, or read into, `b`.
static CUresult anyWrite(uintptr_t p, const void *b, size_t n) {
  if (onDevice(p)) return devWrite((CUdeviceptr)p, b, n);
  memcpy((void *)p, b, n);
  return CUDA_SUCCESS;
}
static CUresult anyRead(void *b, uintptr_t p, size_t n) {
  if (onDevice(p)) return devRead(b, (CUdeviceptr)p, n);
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  memcpy(b, (const void *)p, n);
  return CUDA_SUCCESS;
}

// Rows of `width` elements of `elem` bytes, each `value`.
static CUresult runSet(Op *o) {
  size_t rowBytes = o->width * o->elem;
  if (!onDevice(o->dst)) {  // host memory
    for (size_t y = 0; y < o->height; y++)
      for (size_t i = 0; i < o->width; i++)
        memcpy((void *)(o->dst + y * o->dpitch + i * o->elem), &o->value, o->elem);
    return CUDA_SUCCESS;
  }
  uint8_t b0 = (uint8_t)o->value;
  int uniform = o->elem == 1 || (o->elem == 2 && (o->value & 0xffff) == b0 * 0x0101u) ||
                (o->elem == 4 && o->value == b0 * 0x01010101u);
  if (uniform && (o->dpitch == rowBytes || o->height == 1))
    return devSet(o->dst, b0, rowBytes * o->height);
  unsigned char *row = NULL;
  if (!uniform) {
    row = malloc(rowBytes ? rowBytes : 1);
    for (size_t i = 0; i < o->width; i++) memcpy(row + i * o->elem, &o->value, o->elem);
  }
  CUresult r = CUDA_SUCCESS;
  for (size_t y = 0; y < o->height && r == CUDA_SUCCESS; y++)
    r = uniform ? devSet(o->dst + y * o->dpitch, b0, rowBytes)
                : devWrite(o->dst + y * o->dpitch, row, rowBytes);
  free(row);
  return r;
}

static int valueMet(uint64_t v, Op *o) {
  uint64_t want = o->value64;
  switch (o->flags & 0x3) {
    case CU_STREAM_WAIT_VALUE_GEQ:
      return o->is64 ? (int64_t)(v - want) >= 0 : (int32_t)((uint32_t)v - (uint32_t)want) >= 0;
    case CU_STREAM_WAIT_VALUE_EQ: return v == want;
    case CU_STREAM_WAIT_VALUE_AND: return (v & want) != 0;
    default: return (~(v | want) & (o->is64 ? ~0ull : 0xffffffffull)) != 0;  // NOR, at the word's width
  }
}

static CUresult runOp(Op *o) {
  switch (o->kind) {
    case OP_KERNEL: return runKernel(o);
    case OP_COPY: return runCopy(o);
    case OP_SET: return runSet(o);
    case OP_HOST: o->fn(o->user); return CUDA_SUCCESS;
    case OP_CALLBACK: o->cb(o->stream, sticky, o->user); return CUDA_SUCCESS;
    case OP_RECORD: {
      uint64_t c = devCycles();
      pthread_mutex_lock(&workLock);
      eventMark(o->ev, o->seq, c);
      pthread_mutex_unlock(&workLock);
      return CUDA_SUCCESS;
    }
    case OP_WAIT: {
      // A failed device runs no more work, so the recording may never run: the failure ends the
      // wait.
      pthread_mutex_lock(&workLock);
      while (!eventDone(o->ev, o->seq) && !sticky) pthread_cond_wait(&workCv, &workLock);
      CUresult r = sticky;
      pthread_mutex_unlock(&workLock);
      return r;
    }
    case OP_FREE: return freeRange(o->dst, 0);
    case OP_GRAPH: return runExec(o);
    case OP_WRITE_VALUE: {
      uint64_t v = o->value64;
      return anyWrite(o->dst, &v, o->is64 ? 8 : 4);
    }
    case OP_WAIT_VALUE:
      for (;;) {
        pthread_mutex_lock(&workLock);
        uint64_t seen = epoch;
        pthread_mutex_unlock(&workLock);
        uint64_t v = 0;
        CUresult r = anyRead(&v, o->dst, o->is64 ? 8 : 4);
        if (r != CUDA_SUCCESS) return r;
        if (valueMet(v, o)) return CUDA_SUCCESS;
        pthread_mutex_lock(&workLock);
        while (epoch == seen && !sticky) pthread_cond_wait(&workCv, &workLock);
        CUresult failed = sticky;
        pthread_mutex_unlock(&workLock);
        if (failed) return failed;
      }
    case OP_NOP: return CUDA_SUCCESS;
  }
  return CUDA_ERROR_INVALID_VALUE;
}

// Work done: the device's failure kept, every waiter told.
static void finished(CUresult r) {
  pthread_mutex_lock(&workLock);
  if (r != CUDA_SUCCESS && !sticky) sticky = r;
  epoch++;
  pthread_cond_broadcast(&workCv);
  pthread_mutex_unlock(&workLock);
}

// Dependency collection and completion state are protected by workLock.
static void dependsOn(Op *o, Completion *c) {
  if (!c || c->done) return;
  for (size_t i = 0; i < o->nbefore; i++) if (o->before[i] == c) return;
  completionRetain(c);
  o->before = realloc(o->before, sizeof *o->before * (o->nbefore + 1));
  o->before[o->nbefore++] = c;
}
static void waitBefore(Op *o) {
  for (size_t i = 0; i < o->nbefore; i++)
    while (!o->before[i]->done && !sticky) pthread_cond_wait(&workCv, &workLock);
}
static CUresult runSubmitted(Op *o) {
  pthread_mutex_lock(&workLock);
  waitBefore(o);
  CUresult r = sticky;
  pthread_mutex_unlock(&workLock);
  int was = gcur;
  gcur = o->gpu;
  if (!r || o->kind == OP_CALLBACK) r = runOp(o);
  gcur = was;
  if (o->graph) { graphFree(o->graph); o->graph = NULL; }
  if (o->exec) { execRelease(o->exec); o->exec = NULL; }
  pthread_mutex_lock(&workLock);
  if (r && !sticky) sticky = r;
  o->completion->done = 1;
  epoch++;
  pthread_cond_broadcast(&workCv);
  pthread_mutex_unlock(&workLock);
  return r;
}

static void *streamThread(void *arg) {
  struct CUstream_st *s = arg;
  pthread_mutex_lock(&workLock);
  for (;;) {
    while (!s->head && !s->dying) pthread_cond_wait(&workCv, &workLock);
    if (!s->head) break;
    Op *o = s->head;
    s->running = 1;
    pthread_mutex_unlock(&workLock);
    runSubmitted(o);
    pthread_mutex_lock(&workLock);
    s->head = o->next;
    if (!s->head) s->tail = NULL;
    s->running = 0;
    pthread_cond_broadcast(&workCv);
    pthread_mutex_unlock(&workLock);
    opFree(o);
    pthread_mutex_lock(&workLock);
  }
  for (struct CUstream_st **q = &streams; *q; q = &(*q)->nextStream)
    if (*q == s) {
      *q = s->nextStream;
      break;
    }
  pthread_cond_broadcast(&workCv);
  pthread_mutex_unlock(&workLock);
  completionRelease(s->last);
  free(s->deps);
  free(s);
  return NULL;
}

// Capture and queue insertion share the same submission point as event publication and legacy
// ordering. A recording is never visible before its stream's dependency token is published.
static CUresult submit(CUstream s, Op *o) {
  pthread_mutex_lock(&workLock);
  if (!isLegacy(s) && s->capture) {
    CUresult r = captureOp(s, o);
    pthread_mutex_unlock(&workLock);
    return r;
  }
  if (!isLegacy(s) && s->dying) {
    pthread_mutex_unlock(&workLock);
    opFree(o);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  o->completion = completionNew();
  o->gpu = isLegacy(s) ? gcur : s->gpu;
  if (isLegacy(s)) {
    dependsOn(o, legacyLast);
    for (CUstream t = streams; t; t = t->nextStream)
      if (!(t->flags & CU_STREAM_NON_BLOCKING)) dependsOn(o, t->last);
    completionReplace(&legacyLast, o->completion);
  } else {
    if (!(s->flags & CU_STREAM_NON_BLOCKING)) dependsOn(o, legacyLast);
    completionReplace(&s->last, o->completion);
  }
  if (o->kind == OP_RECORD) o->seq = ++o->ev->recorded;
  if (o->kind == OP_WAIT) o->seq = o->ev->recorded;
  if (o->kind == OP_GRAPH) armLaunch(o);
  if (isLegacy(s)) {
    pthread_mutex_unlock(&workLock);
    CUresult r = runSubmitted(o);
    opFree(o);
    return r;
  }
  o->next = NULL;
  if (s->tail) s->tail->next = o;
  else s->head = o;
  s->tail = o;
  if (!s->started) {
    s->started = 1;
    pthread_create(&s->thread, NULL, streamThread, s);
    pthread_detach(s->thread);
  }
  pthread_cond_broadcast(&workCv);
  CUresult r = sticky;
  pthread_mutex_unlock(&workLock);
  return r;
}

static CUresult synchronize(CUstream s, int context) {
  Op *wait = newOp(OP_NOP);
  pthread_mutex_lock(&workLock);
  if (context || isLegacy(s)) {
    dependsOn(wait, legacyLast);
    for (CUstream t = streams; t; t = t->nextStream)
      if (context || !(t->flags & CU_STREAM_NON_BLOCKING)) dependsOn(wait, t->last);
  } else dependsOn(wait, s->last);
  waitBefore(wait);
  CUresult r = sticky;
  pthread_mutex_unlock(&workLock);
  opFree(wait);
  return r;
}
static CUresult streamSync(CUstream s) { return synchronize(s, 0); }

// ---- Streams ----

CUresult CUDAAPI cuStreamCreate(CUstream *out, unsigned int flags) {
  NEED_INIT();
  struct CUstream_st *s = calloc(1, sizeof *s);
  s->flags = flags;
  s->gpu = gcur;
  pthread_mutex_lock(&workLock);
  s->id = streamIds++;
  s->nextStream = streams;
  streams = s;
  pthread_mutex_unlock(&workLock);
  *out = s;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *s, unsigned int f, int p) {
  CUresult r = cuStreamCreate(s, f);
  if (r == CUDA_SUCCESS) (*s)->priority = p;
  return r;
}
CUresult CUDAAPI cuStreamGetPriority(CUstream s, int *p) {
  *p = isLegacy(s) ? 0 : s->priority;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamGetFlags(CUstream s, unsigned int *f) {
  *f = isLegacy(s) ? 0 : s->flags;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamGetId(CUstream s, unsigned long long *id) {
  *id = isLegacy(s) ? (s == CU_STREAM_PER_THREAD ? 1 : 0) : s->id;
  return CUDA_SUCCESS;
}
static int streamGpu(CUstream s) { return isLegacy(s) ? gcur : s->gpu; }
CUresult CUDAAPI cuStreamGetCtx(CUstream s, CUcontext *c) {
  *c = &gpus[streamGpu(s)].primary;
  return CUDA_SUCCESS;
}
// Green contexts (partitions of a device's SMs) are not modelled: no stream belongs to one.
CUresult CUDAAPI cuStreamGetGreenCtx(CUstream s, CUgreenCtx *g) {
  (void)s;
  *g = NULL;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamGetCtx_v2(CUstream s, CUcontext *c, CUgreenCtx *g) {
  if (c) *c = &gpus[streamGpu(s)].primary;
  if (g) *g = NULL;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamGetDevice(CUstream s, CUdevice *d) {
  *d = streamGpu(s);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamDestroy(CUstream s) {
  if (isLegacy(s)) return CUDA_ERROR_INVALID_HANDLE;
  pthread_mutex_lock(&workLock);
  int started = s->started;
  s->dying = 1;
  // Its thread drains its work, leaves `streams` and frees it; a stream never used leaves here.
  if (!started)
    for (struct CUstream_st **q = &streams; *q; q = &(*q)->nextStream)
      if (*q == s) {
        *q = s->nextStream;
        break;
      }
  pthread_cond_broadcast(&workCv);
  pthread_mutex_unlock(&workLock);
  if (!started) {
    free(s->deps);
    free(s);
  }
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamSynchronize(CUstream s) {
  if (!initialized) return CUDA_ERROR_NOT_INITIALIZED;
  return streamSync(s);
}
CUresult CUDAAPI cuStreamQuery(CUstream s) {
  if (!initialized) return CUDA_ERROR_NOT_INITIALIZED;
  pthread_mutex_lock(&workLock);
  int busy = 0;
  if (isLegacy(s)) {
    busy = legacyLast && !legacyLast->done;
    for (struct CUstream_st *t = streams; t; t = t->nextStream)
      if (t->last && !t->last->done && !(t->flags & CU_STREAM_NON_BLOCKING)) busy = 1;
  } else
    busy = s->last && !s->last->done;
  CUresult r = sticky ? sticky : busy ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
  pthread_mutex_unlock(&workLock);
  return r;
}
CUresult CUDAAPI cuCtxSynchronize(void) {
  if (!initialized) return CUDA_ERROR_NOT_INITIALIZED;
  return synchronize(NULL, 1);
}
CUresult CUDAAPI cuCtxSynchronize_v2(CUcontext c) {
  (void)c;
  return cuCtxSynchronize();
}
CUresult CUDAAPI cuStreamWaitEvent(CUstream s, CUevent e, unsigned int f) {
  NEED_INIT();
  pthread_mutex_lock(&workLock);
  // A wait on an event recorded in a capture still running joins this stream to it (graph.h).
  if (!(f & CU_EVENT_WAIT_EXTERNAL) && e->capture && !isLegacy(s) && !s->capture) {
    struct CUstream_st *origin = NULL;
    for (struct CUstream_st *t = streams; t && !origin; t = t->nextStream)
      if (t->capture == e->capture && t->captureId == e->captureId) origin = t;
    if (origin) {
      s->capture = e->capture;
      s->captureMode = origin->captureMode;
      s->captureId = origin->captureId;
      s->captureOrigin = origin->captureOrigin;
      s->ndeps = 0;
    } else
      e->capture = NULL;  // its capture has ended
  }
  uint64_t seq = e->recorded;
  int capturing = !isLegacy(s) && s->capture;
  if (seq == 0 && !capturing) {
    pthread_mutex_unlock(&workLock);
    return CUDA_SUCCESS;
  }
  eventRetain(e);
  pthread_mutex_unlock(&workLock);
  Op *o = newOp(OP_WAIT);
  o->ev = e;
  o->seq = seq;
  o->flags = f;
  return submit(s, o);
}
CUresult CUDAAPI cuStreamAddCallback(CUstream s, CUstreamCallback cb, void *user, unsigned int f) {
  NEED_INIT();
  (void)f;
  Op *o = newOp(OP_CALLBACK);
  o->cb = cb;
  o->stream = s;
  o->user = user;
  return submit(s, o);
}
CUresult CUDAAPI cuLaunchHostFunc(CUstream s, CUhostFn fn, void *user) {
  NEED_INIT();
  Op *o = newOp(OP_HOST);
  o->fn = fn;
  o->user = user;
  return submit(s, o);
}
CUresult CUDAAPI cuStreamGetAttribute(CUstream s, CUstreamAttrID a, CUstreamAttrValue *v) {
  (void)s;
  memset(v, 0, sizeof *v);
  if (a == CU_LAUNCH_ATTRIBUTE_PRIORITY && !isLegacy(s)) v->priority = s->priority;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamSetAttribute(CUstream s, CUstreamAttrID a, const CUstreamAttrValue *v) {
  if (a == CU_LAUNCH_ATTRIBUTE_PRIORITY && !isLegacy(s)) s->priority = v->priority;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamCopyAttributes(CUstream dst, CUstream src) {
  if (!isLegacy(dst) && !isLegacy(src)) dst->priority = src->priority;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamAttachMemAsync(CUstream s, CUdeviceptr p, size_t n, unsigned int f) {
  (void)s; (void)p; (void)n; (void)f;
  return CUDA_SUCCESS;  // no managed memory: nothing to attach
}

// ---- Events ----

CUresult CUDAAPI cuEventCreate(CUevent *e, unsigned int flags) {
  *e = calloc(1, sizeof(struct CUevent_st));
  (*e)->flags = flags;
  (*e)->refs = 1;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventDestroy(CUevent e) {
  eventRelease(e);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventRecordWithFlags(CUevent e, CUstream s, unsigned int f) {
  NEED_INIT();
  eventRetain(e);
  Op *o = newOp(OP_RECORD);
  o->ev = e;
  o->flags = f;
  return submit(s, o);
}
CUresult CUDAAPI cuEventRecord(CUevent e, CUstream s) {
  return cuEventRecordWithFlags(e, s, 0);
}
CUresult CUDAAPI cuEventSynchronize(CUevent e) {
  pthread_mutex_lock(&workLock);
  uint64_t want = e->recorded;
  while (!eventDone(e, want) && !sticky) pthread_cond_wait(&workCv, &workLock);
  CUresult r = sticky;
  pthread_mutex_unlock(&workLock);
  return r;
}
CUresult CUDAAPI cuEventQuery(CUevent e) {
  pthread_mutex_lock(&workLock);
  CUresult r = sticky ? sticky : !eventDone(e, e->recorded) ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
  pthread_mutex_unlock(&workLock);
  return r;
}
// Simulated time between the events: cycles at the part's clock (timed runs; functional runs
// count issue steps, not time).
CUresult CUDAAPI cuEventElapsedTime(float *ms, CUevent a, CUevent b) {
  if ((a->flags | b->flags) & CU_EVENT_DISABLE_TIMING) return CUDA_ERROR_INVALID_HANDLE;
  pthread_mutex_lock(&workLock);
  int ready = a->recorded && b->recorded && eventDone(a, a->recorded) && eventDone(b, b->recorded) &&
              a->cyclesOf == a->recorded && b->cyclesOf == b->recorded;
  uint64_t ca = a->cycles, cb = b->cycles;
  pthread_mutex_unlock(&workLock);
  if (!ready) return CUDA_ERROR_NOT_READY;
  *ms = (float)(((double)cb - (double)ca) / (G.part->clockMHz * 1000.0));
  return CUDA_SUCCESS;
}

// ---- Copies ----

// Enqueue a copy; with host memory the driver does not know (pageable) on either side, the call
// returns once the copy is done, as the driver's does for pageable memory.
static CUresult enqueueCopy(CUstream s, Op *o) {
  int pageable = (!onDevice(o->dst) && !pinned((void *)o->dst)) ||
                 (!onDevice(o->src) && !pinned((void *)o->src));
  int capturing = 0;
  if (!isLegacy(s)) {
    pthread_mutex_lock(&workLock);
    capturing = s->capture != NULL;
    pthread_mutex_unlock(&workLock);
  }
  CUresult r = submit(s, o);
  if (r == CUDA_SUCCESS && pageable && !capturing) r = streamSync(s);
  return r;
}

static Op *copyOp(uintptr_t dst, uintptr_t src, size_t n) {
  Op *o = newOp(OP_COPY);
  o->dst = dst;
  o->src = src;
  o->width = o->dpitch = o->spitch = o->dslice = o->sslice = n;
  return o;
}

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t n) {
  NEED_INIT();
  return submit(NULL, copyOp(dst, (uintptr_t)src, n));
}
CUresult CUDAAPI cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t n) {
  NEED_INIT();
  return submit(NULL, copyOp((uintptr_t)dst, src, n));
}
CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t n) {
  NEED_INIT();
  return submit(NULL, copyOp(dst, src, n));
}
CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t n) {
  NEED_INIT();
  return submit(NULL, copyOp(dst, src, n));
}
CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dst, CUcontext dc, CUdeviceptr src, CUcontext sc, size_t n) {
  (void)dc; (void)sc;
  return cuMemcpy(dst, src, n);
}
CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t n, CUstream s) {
  NEED_INIT();
  return enqueueCopy(s, copyOp(dst, (uintptr_t)src, n));
}
CUresult CUDAAPI cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t n, CUstream s) {
  NEED_INIT();
  return enqueueCopy(s, copyOp((uintptr_t)dst, src, n));
}
CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s) {
  NEED_INIT();
  return enqueueCopy(s, copyOp(dst, src, n));
}
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s) {
  NEED_INIT();
  return enqueueCopy(s, copyOp(dst, src, n));
}
CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dst, CUcontext dc, CUdeviceptr src, CUcontext sc,
                                   size_t n, CUstream s) {
  (void)dc; (void)sc;
  return cuMemcpyAsync(dst, src, n, s);
}

// A side of a 2D/3D copy: its address, and a CUDA array's own pitch and rows a slice (`tex.h`).
static int sideOf(CUmemorytype t, const void *host, CUdeviceptr d, CUarray a, uintptr_t *out, size_t *pitch,
                  size_t *rows) {
  switch (t) {
    case CU_MEMORYTYPE_HOST: *out = (uintptr_t)host; return 1;
    case CU_MEMORYTYPE_DEVICE: *out = d; return 1;
    case CU_MEMORYTYPE_UNIFIED: *out = d; return 1;
    case CU_MEMORYTYPE_ARRAY:
      if (!a) return 0;
      *out = a->base;
      *pitch = a->pitch;
      *rows = a->height ? a->height : 1;
      return 1;
    default: return 0;
  }
}

// The copy a `CUDA_MEMCPY3D` describes, or NULL when a side is not one. Copies and graph copy nodes
// both take theirs from here.
static Op *copy3DOp(const CUDA_MEMCPY3D *c) {
  uintptr_t dst, src;
  size_t sp = c->srcPitch ? c->srcPitch : c->WidthInBytes, dp = c->dstPitch ? c->dstPitch : c->WidthInBytes;
  size_t sh = c->srcHeight ? c->srcHeight : c->Height, dh = c->dstHeight ? c->dstHeight : c->Height;
  if (!sideOf(c->dstMemoryType, c->dstHost, c->dstDevice, c->dstArray, &dst, &dp, &dh) ||
      !sideOf(c->srcMemoryType, c->srcHost, c->srcDevice, c->srcArray, &src, &sp, &sh))
    return NULL;
  Op *o = newOp(OP_COPY);
  o->src = src + c->srcZ * sp * sh + c->srcY * sp + c->srcXInBytes;
  o->dst = dst + c->dstZ * dp * dh + c->dstY * dp + c->dstXInBytes;
  o->width = c->WidthInBytes;
  o->height = c->Height ? c->Height : 1;
  o->depth = c->Depth ? c->Depth : 1;
  o->spitch = sp;
  o->dpitch = dp;
  o->sslice = sp * sh;
  o->dslice = dp * dh;
  return o;
}

static CUresult copy3D(const CUDA_MEMCPY3D *c, CUstream s, int async) {
  NEED_INIT();
  Op *o = copy3DOp(c);
  if (!o) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return async ? enqueueCopy(s, o) : submit(NULL, o);
}

static void widen(const CUDA_MEMCPY2D *c, CUDA_MEMCPY3D *d) {
  memset(d, 0, sizeof *d);
  d->srcXInBytes = c->srcXInBytes;
  d->srcY = c->srcY;
  d->srcMemoryType = c->srcMemoryType;
  d->srcHost = c->srcHost;
  d->srcDevice = c->srcDevice;
  d->srcArray = c->srcArray;
  d->srcPitch = c->srcPitch;
  d->dstXInBytes = c->dstXInBytes;
  d->dstY = c->dstY;
  d->dstMemoryType = c->dstMemoryType;
  d->dstHost = c->dstHost;
  d->dstDevice = c->dstDevice;
  d->dstArray = c->dstArray;
  d->dstPitch = c->dstPitch;
  d->WidthInBytes = c->WidthInBytes;
  d->Height = c->Height;
  d->Depth = 1;
}

CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *c) {
  CUDA_MEMCPY3D d;
  widen(c, &d);
  return copy3D(&d, NULL, 0);
}
CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *c) { return cuMemcpy2D(c); }
CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *c, CUstream s) {
  CUDA_MEMCPY3D d;
  widen(c, &d);
  return copy3D(&d, s, 1);
}
CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *c) { return copy3D(c, NULL, 0); }
CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D *c, CUstream s) { return copy3D(c, s, 1); }

// ---- Sets ----

static CUresult enqueueSet(CUstream s, CUdeviceptr dst, uint32_t v, unsigned elem, size_t width,
                           size_t height, size_t pitch) {
  NEED_INIT();
  Op *o = newOp(OP_SET);
  o->dst = dst;
  o->value = v;
  o->elem = elem;
  o->width = width;
  o->height = height;
  o->dpitch = pitch ? pitch : width * elem;
  return submit(s, o);
}

CUresult CUDAAPI cuMemsetD8(CUdeviceptr d, unsigned char v, size_t n) { return enqueueSet(NULL, d, v, 1, n, 1, 0); }
CUresult CUDAAPI cuMemsetD16(CUdeviceptr d, unsigned short v, size_t n) { return enqueueSet(NULL, d, v, 2, n, 1, 0); }
CUresult CUDAAPI cuMemsetD32(CUdeviceptr d, unsigned int v, size_t n) { return enqueueSet(NULL, d, v, 4, n, 1, 0); }
CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream s) { return enqueueSet(s, d, v, 1, n, 1, 0); }
CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr d, unsigned short v, size_t n, CUstream s) { return enqueueSet(s, d, v, 2, n, 1, 0); }
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr d, unsigned int v, size_t n, CUstream s) { return enqueueSet(s, d, v, 4, n, 1, 0); }
CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr d, size_t p, unsigned char v, size_t w, size_t h) { return enqueueSet(NULL, d, v, 1, w, h, p); }
CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr d, size_t p, unsigned short v, size_t w, size_t h) { return enqueueSet(NULL, d, v, 2, w, h, p); }
CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr d, size_t p, unsigned int v, size_t w, size_t h) { return enqueueSet(NULL, d, v, 4, w, h, p); }
CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr d, size_t p, unsigned char v, size_t w, size_t h, CUstream s) { return enqueueSet(s, d, v, 1, w, h, p); }
CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr d, size_t p, unsigned short v, size_t w, size_t h, CUstream s) { return enqueueSet(s, d, v, 2, w, h, p); }
CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr d, size_t p, unsigned int v, size_t w, size_t h, CUstream s) { return enqueueSet(s, d, v, 4, w, h, p); }

// ---- Stream-ordered allocation: memory is taken at once (its address is the program's from the
// call), and given back in stream order ----

CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *p, size_t n, CUstream s) {
  NEED_INIT();
  pthread_mutex_lock(&workLock);
  int capturing = !isLegacy(s) && s->capture;
  CUresult r = capturing ? captureAlloc(s, p, n) : CUDA_SUCCESS;
  pthread_mutex_unlock(&workLock);
  return capturing ? r : allocRange(p, n, ALLOC_ALIGN, 0);
}
CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr *p, size_t n, CUmemoryPool pool, CUstream s) {
  (void)pool;
  return cuMemAllocAsync(p, n, s);
}
CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr p, CUstream s) {
  NEED_INIT();
  Op *o = newOp(OP_FREE);
  o->dst = p;
  return submit(s, o);
}

// ---- Stream memory operations ----

static CUresult valueOp(CUstream s, OpKind k, CUdeviceptr addr, uint64_t v, unsigned flags, int is64) {
  NEED_INIT();
  Op *o = newOp(k);
  o->dst = addr;
  o->value64 = v;
  o->flags = flags;
  o->is64 = is64;
  return submit(s, o);
}
CUresult CUDAAPI cuStreamWriteValue32(CUstream s, CUdeviceptr a, cuuint32_t v, unsigned int f) { return valueOp(s, OP_WRITE_VALUE, a, v, f, 0); }
CUresult CUDAAPI cuStreamWriteValue64(CUstream s, CUdeviceptr a, cuuint64_t v, unsigned int f) { return valueOp(s, OP_WRITE_VALUE, a, v, f, 1); }
CUresult CUDAAPI cuStreamWaitValue32(CUstream s, CUdeviceptr a, cuuint32_t v, unsigned int f) { return valueOp(s, OP_WAIT_VALUE, a, v, f, 0); }
CUresult CUDAAPI cuStreamWaitValue64(CUstream s, CUdeviceptr a, cuuint64_t v, unsigned int f) { return valueOp(s, OP_WAIT_VALUE, a, v, f, 1); }
CUresult CUDAAPI cuStreamBatchMemOp(CUstream s, unsigned int n, CUstreamBatchMemOpParams *ps, unsigned int f) {
  (void)f;
  for (unsigned i = 0; i < n; i++) {
    CUstreamBatchMemOpParams *p = &ps[i];
    CUresult r = CUDA_SUCCESS;
    switch (p->operation) {
      case CU_STREAM_MEM_OP_WAIT_VALUE_32:
        r = valueOp(s, OP_WAIT_VALUE, p->waitValue.address, p->waitValue.value, p->waitValue.flags, 0);
        break;
      case CU_STREAM_MEM_OP_WAIT_VALUE_64:
        r = valueOp(s, OP_WAIT_VALUE, p->waitValue.address, p->waitValue.value64, p->waitValue.flags, 1);
        break;
      case CU_STREAM_MEM_OP_WRITE_VALUE_32:
        r = valueOp(s, OP_WRITE_VALUE, p->writeValue.address, p->writeValue.value, p->writeValue.flags, 0);
        break;
      case CU_STREAM_MEM_OP_WRITE_VALUE_64:
        r = valueOp(s, OP_WRITE_VALUE, p->writeValue.address, p->writeValue.value64, p->writeValue.flags, 1);
        break;
      default: break;  // flushes: every write is visible at once here
    }
    if (r != CUDA_SUCCESS) return r;
  }
  return CUDA_SUCCESS;
}

// ---- Launches ----

// The launch's work, its parameter block laid out as the kernel's ELF says, from `args` (one
// pointer per parameter) or `extra` (a packed buffer).
static CUresult kernelOp(Op **out, CUfunction f, const unsigned grid[3], const unsigned block[3],
                         const unsigned cluster[3], unsigned smem, void **args, void **extra) {
  if (!f) return CUDA_ERROR_INVALID_HANDLE;
  CUresult r = resolve(f);
  if (r != CUDA_SUCCESS) return r;
  unsigned threads = block[0] * block[1] * block[2];
  if (!grid[0] || !grid[1] || !grid[2] || !threads) return CUDA_ERROR_INVALID_VALUE;
  if ((int)threads > maxThreadsOf(f)) {
    note("%s: %u threads per block, more than its %d", f->name, threads, maxThreadsOf(f));
    return CUDA_ERROR_INVALID_VALUE;
  }
  int dynMax = f->maxDynShared >= 0 ? f->maxDynShared : 48 * 1024 - f->shared;
  if ((int)smem > dynMax) {
    note("%s: %u bytes of dynamic shared memory, more than its %d", f->name, smem, dynMax);
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (int i = 0; i < 3; i++)
    if (cluster[i] && grid[i] % cluster[i]) {
      note("%s: grid not a whole number of clusters", f->name);
      return CUDA_ERROR_INVALID_CLUSTER_SIZE;
    }
  // A parameter the kernel's parameter block does not hold is a layout the device misread: refused,
  // not launched with the parameter left as zeros.
  for (uint32_t i = 0; args && i < f->nparams; i++)
    if (f->poff[i] + f->psize[i] > f->paramBytes) {
      note("%s: parameter %u (%u bytes at offset %u) outside its %u-byte parameter block", f->name, i,
           f->psize[i], f->poff[i], f->paramBytes);
      return CUDA_ERROR_INVALID_VALUE;
    }
  Op *o = newOp(OP_KERNEL);
  o->f = f;
  memcpy(o->grid, grid, sizeof o->grid);
  memcpy(o->block, block, sizeof o->block);
  for (int i = 0; i < 3; i++) o->cluster[i] = cluster[i] ? cluster[i] : 1;
  o->smem = smem;
  o->paramBytes = f->paramBytes;
  o->params = calloc(1, f->paramBytes ? f->paramBytes : 1);
  if (args) {
    for (uint32_t i = 0; i < f->nparams; i++) memcpy(o->params + f->poff[i], args[i], f->psize[i]);
  } else if (extra) {
    void *buf = NULL;
    size_t sz = 0;
    for (int i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
      if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = extra[i + 1];
      if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) sz = *(size_t *)extra[i + 1];
    }
    if (buf) memcpy(o->params, buf, sz < f->paramBytes ? sz : f->paramBytes);
  }
  *out = o;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                                unsigned by, unsigned bz, unsigned smem, CUstream s, void **args,
                                void **extra) {
  NEED_INIT();
  unsigned g[3] = {gx, gy, gz}, b[3] = {bx, by, bz}, c[3] = {0, 0, 0};
  Op *o;
  CUresult r = kernelOp(&o, f, g, b, c, smem, args, extra);
  return r == CUDA_SUCCESS ? submit(s, o) : r;
}

CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                           unsigned bx, unsigned by, unsigned bz, unsigned smem,
                                           CUstream s, void **args) {
  return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, s, args, NULL);
}

// A launch with attributes: the cluster shape; events recorded when the kernel is done (the
// programmatic event and the launch-completion event: the kernel's blocks have all started and
// finished by then). The scheduling hints (priority, memory synchronization domains, access
// policy, programmatic serialization) change no result of a run that executes work in order.
CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *c, CUfunction f, void **args, void **extra) {
  NEED_INIT();
  unsigned g[3] = {c->gridDimX, c->gridDimY, c->gridDimZ};
  unsigned b[3] = {c->blockDimX, c->blockDimY, c->blockDimZ};
  unsigned cl[3] = {0, 0, 0};
  CUevent after[2] = {NULL, NULL};
  for (unsigned i = 0; i < c->numAttrs; i++) {
    const CUlaunchAttribute *a = &c->attrs[i];
    switch (a->id) {
      case CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:
        cl[0] = a->value.clusterDim.x;
        cl[1] = a->value.clusterDim.y;
        cl[2] = a->value.clusterDim.z;
        break;
      case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT: after[0] = a->value.programmaticEvent.event; break;
#if CUDA_VERSION >= 12030
      case CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT: after[1] = a->value.launchCompletionEvent.event; break;
#endif
      default: break;
    }
  }
  Op *o;
  CUresult r = kernelOp(&o, f, g, b, cl, c->sharedMemBytes, args, extra);
  if (r != CUDA_SUCCESS) return r;
  r = submit(c->hStream, o);
  for (int i = 0; i < 2 && r == CUDA_SUCCESS; i++)
    if (after[i]) r = cuEventRecord(after[i], c->hStream);
  return r;
}
