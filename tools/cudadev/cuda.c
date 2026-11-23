// libcuda.so.1 for the emulated device: the CUDA driver API over the nvgpu-lean simulator.
//
// Unproven glue (outside the proven library): it keeps the driver's bookkeeping (contexts,
// allocations, modules, streams, events, graphs) and hands memory traffic, module loading and
// kernel launches to the Lean side's C ABI (`Device/Api.lean`). One translation unit: this file
// (initialization, devices, contexts, memory, errors, entry points) and the parts it includes
// (module.h: fat binaries, modules and libraries; stream.h: streams, events, copies and launches;
// graph.h: graphs and stream capture; tmap.h: tensor maps; runtime_tables.h: the runtime's tables).
//
// Environment: NVGPU_LEAN_GPU (h200 | h100 | h100-pcie | h100-nvl | a100-40 | a100-80 | b200 |
// rtxpro6000; default h200), NVGPU_LEAN_TIMED=1 (cycle-level runs; default functional),
// NVGPU_LEAN_CUDA (the toolkit whose cuobjdump/ptxas to use; default /usr/local/cuda),
// NVGPU_LEAN_CUOBJDUMP_OLD (a CUDA 12 cuobjdump, for cubins in the ELF format before CUDA 13's),
// NVGPU_LEAN_CUOBJDUMP_NEW (a newer cuobjdump, for cubins newer than the toolkit's reads),
// NVGPU_LEAN_CACHE (where loaded modules are kept; default $XDG_CACHE_HOME/nvgpu-lean).

#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <lean/lean.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "parts.h"
#include "state.h"

// ---- The Lean side ----

extern void lean_initialize_runtime_module(void);
// The task manager the engine's worker threads run on (Lean's own executables start it in main).
extern void lean_init_task_manager(void);
// A thread Lean did not create, before its first call into Lean (not declared in lean.h).
extern void lean_initialize_thread(void);
// The module initializer of Device.Api (its name, which lake's package prefix enters, from build.sh).
extern lean_object *LEAN_INIT(uint8_t builtin, lean_object *w);
extern lean_object *nvgpu_device_create(lean_obj_arg name, uint64_t base, uint64_t bytes,
                                        lean_object *w);
extern lean_object *nvgpu_write(lean_obj_arg d, uint64_t addr, lean_obj_arg data);
extern lean_object *nvgpu_read(lean_obj_arg d, uint64_t addr, uint64_t n, lean_object *w);
extern lean_object *nvgpu_memset(lean_obj_arg d, uint64_t addr, uint8_t v, uint64_t n);
extern lean_object *nvgpu_copy(lean_obj_arg d, uint64_t dst, uint64_t src, uint64_t n);
extern lean_object *nvgpu_release(lean_obj_arg d, uint64_t addr, uint64_t n);
extern lean_object *nvgpu_module_open(lean_obj_arg d, lean_obj_arg sass, lean_obj_arg elf,
                                      uint64_t base);
extern uint64_t nvgpu_module_data_end(lean_obj_arg d, uint32_t m, uint64_t base);
extern lean_object *nvgpu_module_kernel(lean_obj_arg d, uint32_t m, lean_obj_arg name);
extern uint64_t nvgpu_module_global(lean_obj_arg d, uint32_t m, lean_obj_arg name, uint32_t what);
extern uint32_t nvgpu_kernel_count(lean_obj_arg d);
extern uint64_t nvgpu_kernel_info(lean_obj_arg d, uint32_t k, uint32_t what);
extern uint64_t nvgpu_kernel_attr(lean_obj_arg d, uint32_t k, uint32_t what);
extern lean_object *nvgpu_launch(lean_obj_arg d, uint32_t k, uint32_t gx, uint32_t gy, uint32_t gz,
                                 uint32_t bx, uint32_t by, uint32_t bz, uint32_t cx, uint32_t cy,
                                 uint32_t cz, uint32_t smem, lean_obj_arg params,
                                 uint8_t functional, lean_object *w);
extern lean_object *nvgpu_set_textures(lean_obj_arg d, lean_obj_arg table);
extern lean_object *nvgpu_launch_mesh(lean_obj_arg ds, uint32_t g, uint32_t k, uint32_t gx, uint32_t gy,
                                      uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz, uint32_t cx,
                                      uint32_t cy, uint32_t cz, uint32_t smem, lean_obj_arg params,
                                      uint8_t functional, lean_object *w);
extern uint64_t nvgpu_device_cycles(lean_obj_arg d);
extern uint64_t nvgpu_device_info(lean_obj_arg d, uint32_t what);
extern lean_object *nvgpu_device_wait(lean_obj_arg d, uint64_t until);
extern uint64_t nvgpu_collective_ns(lean_obj_arg d, uint64_t kind, uint64_t ranks, uint64_t bytes);
extern lean_object *nvgpu_error_string(lean_obj_arg e);
extern lean_object *nvgpu_tensor_map(lean_obj_arg args);
extern lean_object *nvgpu_tensor_map_address(lean_obj_arg map, uint64_t addr);

// ---- State ----

#define DEVICE_BASE 0x100000000000ull  // device virtual addresses start here (16 TiB)
// Device g's addresses from DEVICE_BASE + g * the stride: the largest device's memory and module
// data, in 64 MB steps, so the devices' memories lie end to end (a mesh launch's global memory is
// all of them, `nvgpu_launch_mesh`).
static uint64_t deviceStride;
#define MODULE_BYTES (16ull << 30)      // module data, past the part's memory (ours)
#define ALLOC_ALIGN 512                 // cuMemAlloc's alignment (the driver guarantees 256)
#define MAX_GPUS 16

// Guards the Lean device and the bookkeeping. Recursive: a call may enter another (cuMemFree
// within a stream's free). Never held while waiting for a stream.
static pthread_mutex_t lock = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static int initialized;
struct CUctx_st {
  int device;
  int refs;
  unsigned flags;
};
typedef struct Alloc Alloc;
// One emulated device: its part, the Lean device (the one reference, updated by each call), its
// memory window's allocations and module region, and its primary context. Every call works on
// the calling thread's device (`gcur`): its current context's, or, for an address, the device
// whose window holds it.
typedef struct {
  const Part *part;
  lean_object *lean;
  Alloc *allocs;
  size_t inUse;
  CUdeviceptr moduleNext;  // the module region's next free byte (past the part's memory)
  uint32_t modulesOpened;  // the Lean side's module count
  struct CUctx_st primary;
} Gpu;
static Gpu gpus[MAX_GPUS];
static int ngpus;
static __thread int gcur;
#define G (gpus[gcur])  // the calling thread's device
static CUdeviceptr gpuBase(int g) { return DEVICE_BASE + (CUdeviceptr)g * deviceStride; }
// The device whose window holds `x`, or -1.
static int gpuAt(CUdeviceptr x) {
  if (x < DEVICE_BASE || deviceStride == 0) return -1;
  CUdeviceptr g = (x - DEVICE_BASE) / deviceStride;
  if ((int)g >= ngpus || x - gpuBase((int)g) >= gpus[g].part->memBytes + MODULE_BYTES) return -1;
  return (int)g;
}
static _Atomic(CUresult) sticky;      // a failed launch or copy: the device is lost, as a GPU's context would be
static int functional = 1;
static char toolkit[512] = "/usr/local/cuda";
static char cacheDir[512];
static char lastError[8192];  // a CUTLASS kernel's mangled name alone passes 1500 bytes
static State *state;   // what the management library of other processes reads (state.h)
static StateProc *me;
static __thread int leanThread;  // this thread may call into Lean

static void note(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(lastError, sizeof lastError, fmt, ap);
  va_end(ap);
  if (getenv("NVGPU_LEAN_VERBOSE")) fprintf(stderr, "nvgpu-lean: %s\n", lastError);
}

// Query values matter as well as status codes. Log direct dlsym callers too, under the same
// diagnostic switch as entry-point tracing, without changing the thread's last error.
static void queryTrace(const char *fmt, ...) {
  const char *mode = getenv("NVGPU_LEAN_TRACE");
  if (!mode || mode[0] != '2') return;
  va_list ap;
  va_start(ap, fmt);
  flockfile(stderr);
  fputs("nvgpu-lean: query ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  funlockfile(stderr);
  va_end(ap);
}

// Before a call into Lean (the lock held): a thread Lean did not create registers once.
static void enterLean(void) {
  if (!leanThread) {
    lean_initialize_thread();
    leanThread = 1;
  }
}

// An IO result: the value on success, else the error noted and NULL.
static lean_object *ioValue(lean_object *r) {
  if (lean_io_result_is_ok(r)) {
    lean_object *v = lean_io_result_get_value(r);
    lean_inc(v);
    lean_dec(r);
    return v;
  }
  lean_object *e = lean_io_result_get_error(r);
  lean_inc(e);
  lean_object *s = nvgpu_error_string(e);
  note("%s", lean_string_cstr(s));
  lean_dec(s);
  lean_dec(r);
  return NULL;
}

// A device-changing call's result (`Device × String`): the device kept, and whether the message
// is empty (the message noted otherwise).
static int take(lean_object *pair) {
  G.lean = lean_ctor_get(pair, 0);
  lean_inc(G.lean);
  lean_object *msg = lean_ctor_get(pair, 1);
  int ok = lean_string_size(msg) == 1;
  if (!ok) note("%s", lean_string_cstr(msg));
  lean_dec(pair);
  return ok;
}

// A mesh call's (devices, message) taken back: each device's object, and the message (empty
// on success, else noted). Every device is lost if the call failed outright.
static int takeMesh(lean_object *pair) {
  if (!pair) {
    for (int d = 0; d < ngpus; d++) gpus[d].lean = NULL;
    return 0;
  }
  lean_object *ds = lean_ctor_get(pair, 0);
  for (int d = 0; d < ngpus; d++) {
    gpus[d].lean = lean_array_get_core(ds, d);
    lean_inc(gpus[d].lean);
  }
  lean_object *msg = lean_ctor_get(pair, 1);
  int ok = lean_string_size(msg) == 1;
  if (!ok) note("%s", lean_string_cstr(msg));
  lean_dec(pair);
  return ok;
}

// The device lent to a query (the lock held).
static lean_object *lend(void) {
  enterLean();
  lean_inc(G.lean);
  return G.lean;
}

// Publish this process's memory in use.
static void publish(int64_t delta) {
  if (!state) return;
  __atomic_add_fetch(&state->memUsed, (uint64_t)delta, __ATOMIC_RELAXED);
  if (me) me->memUsed += delta;
}

// ---- Device memory primitives (the lock taken inside) ----

// Accelerated and resident (accel.h): device memory the host's GPU holds made current here before
// the host reads it, and what the host writes here sent there at the next launch.
static int resPull(CUdeviceptr a, size_t n);
static int resWrote(CUdeviceptr a, size_t n, int before);

// The lock held: the calling thread on device `g` until `gpuLeave`.
static int gpuEnter(int g) { int was = gcur; if (g >= 0) gcur = g; return was; }
static void gpuLeave(int was) { gcur = was; }

static CUresult devWrite(CUdeviceptr dst, const void *src, size_t n) {
  if (n == 0) return CUDA_SUCCESS;
  pthread_mutex_lock(&lock);
  int was = gpuEnter(gpuAt(dst));
  enterLean();
  CUresult r = CUDA_ERROR_ILLEGAL_ADDRESS;
  if (resWrote(dst, n, 1)) {
    lean_object *b = lean_alloc_sarray(1, n, n);
    memcpy(lean_sarray_cptr(b), src, n);
    if (take(nvgpu_write(G.lean, dst, b)) && resWrote(dst, n, 0)) r = CUDA_SUCCESS;
  }
  gpuLeave(was);
  pthread_mutex_unlock(&lock);
  return r;
}

static CUresult devRead(void *dst, CUdeviceptr src, size_t n) {
  if (n == 0) return CUDA_SUCCESS;
  pthread_mutex_lock(&lock);
  int was = gpuEnter(gpuAt(src));
  lean_object *b = resPull(src, n) ? ioValue(nvgpu_read(lend(), src, n, lean_io_mk_world())) : NULL;
  gpuLeave(was);
  CUresult r = CUDA_ERROR_ILLEGAL_ADDRESS;
  if (b) {
    memcpy(dst, lean_sarray_cptr(b), n);
    lean_dec(b);
    r = CUDA_SUCCESS;
  }
  pthread_mutex_unlock(&lock);
  return r;
}

// Device memory through the host, for the emulated NCCL (nccl.c): from a stream's host function,
// where the API's copies (ordered on a stream) would wait on that stream.
CUresult nvgpuLeanRead(void *dst, CUdeviceptr src, size_t n) { return devRead(dst, src, n); }
CUresult nvgpuLeanWrite(CUdeviceptr dst, const void *src, size_t n) { return devWrite(dst, src, n); }

// A mesh's time, for the emulated NCCL: device g's clock in ns (the cycles it has run, at its
// clock), the device waiting until a time (a collective's end), and a collective's time on its
// links (`Device.collectiveNs`).
uint64_t nvgpuLeanDeviceNs(int g) {
  pthread_mutex_lock(&lock);
  int was = gpuEnter(g);
  uint64_t c = nvgpu_device_cycles(lend()), mhz = nvgpu_device_info(lend(), 1);
  gpuLeave(was);
  pthread_mutex_unlock(&lock);
  return mhz ? c * 1000 / mhz : 0;
}
void nvgpuLeanDeviceWaitNs(int g, uint64_t ns) {
  pthread_mutex_lock(&lock);
  int was = gpuEnter(g);
  uint64_t mhz = nvgpu_device_info(lend(), 1);
  G.lean = nvgpu_device_wait(G.lean, (ns * mhz + 999) / 1000);
  gpuLeave(was);
  pthread_mutex_unlock(&lock);
}
uint64_t nvgpuLeanCollectiveNs(int g, int kind, int ranks, uint64_t bytes) {
  pthread_mutex_lock(&lock);
  int was = gpuEnter(g);
  uint64_t ns = nvgpu_collective_ns(lend(), (uint64_t)kind, (uint64_t)ranks, bytes);
  gpuLeave(was);
  pthread_mutex_unlock(&lock);
  return ns;
}

static CUresult devCopy(CUdeviceptr dst, CUdeviceptr src, size_t n) {
  if (n == 0) return CUDA_SUCCESS;
  // Between devices (a peer copy): through the host.
  if (gpuAt(dst) != gpuAt(src)) {
    void *b = malloc(n);
    CUresult r = b ? devRead(b, src, n) : CUDA_ERROR_OUT_OF_MEMORY;
    if (r == CUDA_SUCCESS) r = devWrite(dst, b, n);
    free(b);
    return r;
  }
  pthread_mutex_lock(&lock);
  int was = gpuEnter(gpuAt(dst));
  enterLean();
  CUresult r = CUDA_ERROR_ILLEGAL_ADDRESS;
  if (resPull(src, n) && resWrote(dst, n, 1) && take(nvgpu_copy(G.lean, dst, src, n)) && resWrote(dst, n, 0))
    r = CUDA_SUCCESS;
  gpuLeave(was);
  pthread_mutex_unlock(&lock);
  return r;
}

static CUresult devSet(CUdeviceptr dst, uint8_t v, size_t n) {
  if (n == 0) return CUDA_SUCCESS;
  pthread_mutex_lock(&lock);
  int was = gpuEnter(gpuAt(dst));
  enterLean();
  CUresult r = CUDA_ERROR_ILLEGAL_ADDRESS;
  if (resWrote(dst, n, 1) && take(nvgpu_memset(G.lean, dst, v, n)) && resWrote(dst, n, 0)) r = CUDA_SUCCESS;
  gpuLeave(was);
  pthread_mutex_unlock(&lock);
  return r;
}

static uint64_t devCycles(void) {
  pthread_mutex_lock(&lock);
  uint64_t c = initialized ? nvgpu_device_cycles(lend()) : 0;
  pthread_mutex_unlock(&lock);
  return c;
}

// With NVGPU_LEAN_VERBOSE, at exit: each device's simulated time, the cycles its kernels ran and
// the waits its collectives added (a mesh's step time is the largest).
static void reportClocks(void) {
  for (int d = 0; d < ngpus; d++) {
    uint64_t ns = nvgpuLeanDeviceNs(d);
    fprintf(stderr, "nvgpu-lean: device %d (%s): %.3f ms simulated\n", d, gpus[d].part->name, ns / 1e6);
  }
}

// ---- Initialization ----

CUresult CUDAAPI cuInit(unsigned int flags) {
  (void)flags;
  pthread_mutex_lock(&lock);
  if (!initialized) {
    // NVGPU_LEAN_GPU: a part (`h100`), several (`h100,h100`), or a part times a count (`b200x8`),
    // all of one target (a module is prepared once, for it).
    const char *g = getenv("NVGPU_LEAN_GPU");
    const Part *ps[MAX_GPUS];
    ngpus = partsOf(g, ps, MAX_GPUS);
    for (int d = 0; d < ngpus; d++) {
      if (strcmp(ps[d]->target, ps[0]->target) != 0) ngpus = 0;
      else {
        gpus[d].part = ps[d];
        gpus[d].primary.device = d;
      }
    }
    if (ngpus == 0) {
      pthread_mutex_unlock(&lock);
      return CUDA_ERROR_NO_DEVICE;
    }
    const char *t = getenv("NVGPU_LEAN_TIMED");
    functional = !(t && t[0] == '1');
    const char *tk = getenv("NVGPU_LEAN_CUDA");
    if (tk) snprintf(toolkit, sizeof toolkit, "%s", tk);
    const char *cd = getenv("NVGPU_LEAN_CACHE"), *xdg = getenv("XDG_CACHE_HOME");
    if (cd) snprintf(cacheDir, sizeof cacheDir, "%s", cd);
    else if (xdg) snprintf(cacheDir, sizeof cacheDir, "%s/nvgpu-lean", xdg);
    else snprintf(cacheDir, sizeof cacheDir, "%s/.cache/nvgpu-lean", getenv("HOME") ? getenv("HOME") : ".");
    lean_initialize_runtime_module();
    lean_object *r = LEAN_INIT(1, lean_io_mk_world());
    if (!lean_io_result_is_ok(r)) {
      pthread_mutex_unlock(&lock);
      return CUDA_ERROR_NOT_INITIALIZED;
    }
    lean_dec(r);
    lean_io_mark_end_initialization();
    lean_init_task_manager();
    leanThread = 1;
    for (int d = 0; d < ngpus; d++) {
      uint64_t span = (gpus[d].part->memBytes + MODULE_BYTES + (64ull << 20) - 1) / (64ull << 20) * (64ull << 20);
      if (span > deviceStride) deviceStride = span;
    }
    for (int d = 0; d < ngpus; d++) {
      gpus[d].lean = ioValue(nvgpu_device_create(lean_mk_string(gpus[d].part->name), gpuBase(d), deviceStride,
                                                 lean_io_mk_world()));
      if (!gpus[d].lean) {
        pthread_mutex_unlock(&lock);
        return CUDA_ERROR_NOT_INITIALIZED;
      }
      gpus[d].moduleNext = gpuBase(d) + gpus[d].part->memBytes;
    }
    if (getenv("NVGPU_LEAN_VERBOSE")) atexit(reportClocks);
    state = stateMap(1);
    if (state) {
      if (state->magic != STATE_MAGIC) {
        memset(state, 0, sizeof *state);
        state->magic = STATE_MAGIC;
      }
      snprintf(state->partName, sizeof state->partName, "%s", g && *g ? g : "h200");
      for (int i = 0; i < STATE_PROCS && !me; i++)
        if (state->procs[i].pid == 0 || kill(state->procs[i].pid, 0) != 0) {
          state->procs[i].pid = getpid();
          state->procs[i].memUsed = 0;
          me = &state->procs[i];
        }
    }
    initialized = 1;
  }
  pthread_mutex_unlock(&lock);
  return CUDA_SUCCESS;
}

#define NEED_INIT() \
  do {                \
    if (!initialized) return CUDA_ERROR_NOT_INITIALIZED; \
    if (sticky) return sticky; \
  } while (0)

CUresult CUDAAPI cuDriverGetVersion(int *v) {
  if (!v) return CUDA_ERROR_INVALID_VALUE;
  *v = CUDA_VERSION;
  return CUDA_SUCCESS;
}

// ---- Devices ----

CUresult CUDAAPI cuDeviceGetCount(int *n) {
  NEED_INIT();
  *n = ngpus;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *d, int ordinal) {
  NEED_INIT();
  if (ordinal < 0 || ordinal >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  *d = ordinal;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice d) {
  NEED_INIT();
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  snprintf(name, len, "%s (nvgpu-lean)", gpus[d].part->marketing);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid *u, CUdevice d) {
  NEED_INIT();
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  memset(u, 0, sizeof *u);
  memcpy(u->bytes, "nvgpu-lean-emu", 14);
  u->bytes[15] = (char)d;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetLuid(char *luid, unsigned int *mask, CUdevice d) {
  (void)luid; (void)mask; (void)d;
  return CUDA_ERROR_NOT_SUPPORTED;  // Windows only
}

CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice d) {
  NEED_INIT();
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  *bytes = gpus[d].part->memBytes;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice d) {
  NEED_INIT();
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  *major = gpus[d].part->ccMajor;
  *minor = gpus[d].part->ccMinor;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetAttribute(int *v, CUdevice_attribute a, CUdevice d) {
  NEED_INIT();
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  int x;
  if (!partAttribute(gpus[d].part, a, &x)) return CUDA_ERROR_INVALID_VALUE;
  if (a == CU_DEVICE_ATTRIBUTE_PCI_BUS_ID) x += d;  // each device on its own bus
  *v = x;
  queryTrace("cuDeviceGetAttribute(%d) = %d", (int)a, x);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetPCIBusId(char *id, int len, CUdevice d) {
  NEED_INIT();
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  snprintf(id, len, "0000:%02x:00.0", gpus[d].part->pciBus + d);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *d, const char *id) {
  NEED_INIT();
  unsigned dom = 0, bus = 0;
  if (!id || sscanf(id, "%x:%x", &dom, &bus) != 2) return CUDA_ERROR_INVALID_VALUE;
  for (int g = 0; g < ngpus; g++)
    if ((unsigned)(gpus[g].part->pciBus + g) == bus) {
      *d = g;
      return CUDA_SUCCESS;
    }
  return CUDA_ERROR_INVALID_DEVICE;
}

// Peers: every pair of devices, as on an NVLink/NVSwitch board. Copies between them are
// stream-ordered work; a kernel reaches a peer's memory through its pointers (`nvgpu_launch_mesh`).
CUresult CUDAAPI cuDeviceCanAccessPeer(int *can, CUdevice d, CUdevice peer) {
  NEED_INIT();
  if (d < 0 || d >= ngpus || peer < 0 || peer >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  *can = d != peer;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetP2PAttribute(int *v, CUdevice_P2PAttribute a, CUdevice s, CUdevice d) {
  NEED_INIT();
  if (s < 0 || s >= ngpus || d < 0 || d >= ngpus || s == d) return CUDA_ERROR_INVALID_DEVICE;
  *v = a == CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK ? 0 : 1;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetTexture1DLinearMaxWidth(size_t *w, CUarray_format f, unsigned n, CUdevice d) {
  (void)f; (void)n; (void)d;
  *w = (size_t)1 << 28;
  return CUDA_SUCCESS;
}

// ---- Contexts: each device's primary context; a created context is its device's primary one ----

static __thread CUcontext current;
static void setCurrent(CUcontext c) {
  current = c;
  gcur = c ? c->device : 0;
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *c, CUdevice d) {
  NEED_INIT();
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  gpus[d].primary.refs++;
  *c = &gpus[d].primary;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice d) {
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  if (gpus[d].primary.refs > 0) gpus[d].primary.refs--;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice d) { (void)d; return CUDA_SUCCESS; }
CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice d, unsigned int f) {
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  gpus[d].primary.flags = f;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice d, unsigned int *flags, int *active) {
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  *flags = gpus[d].primary.flags;
  *active = gpus[d].primary.refs > 0;
  return CUDA_SUCCESS;
}

static CUresult legacyCtxCreate(CUcontext *c, unsigned int flags, CUdevice d) {
  NEED_INIT();
  (void)flags;
  if (d < 0 || d >= ngpus) return CUDA_ERROR_INVALID_DEVICE;
  gpus[d].primary.refs++;
  *c = &gpus[d].primary;
  setCurrent(*c);
  return CUDA_SUCCESS;
}
// CUDA 11.4's form, with execution affinity (ignored).
static CUresult legacyCtxCreate_v3(CUcontext *c, CUexecAffinityParam *p, int n, unsigned int flags, CUdevice d) {
  (void)p; (void)n;
  return legacyCtxCreate(c, flags, d);
}
// CUDA 12.5 adds creation parameters (`cuCtxCreate_v4`), which the device ignores.
CUresult CUDAAPI cuCtxCreate_v4(CUcontext *c, CUctxCreateParams *params, unsigned int flags, CUdevice d) {
  (void)params;
  return legacyCtxCreate(c, flags, d);
}
#if CUDA_VERSION < 13000
CUresult CUDAAPI cuCtxCreate(CUcontext *c, unsigned int flags, CUdevice d) { return legacyCtxCreate(c, flags, d); }
#endif
CUresult CUDAAPI cuCtxDestroy(CUcontext c) {
  if (c == current) setCurrent(NULL);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxSetCurrent(CUcontext c) {
  setCurrent(c);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *c) {
  *c = current;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxPushCurrent(CUcontext c) {
  setCurrent(c);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *c) {
  if (c) *c = current;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetDevice(CUdevice *d) {
  *d = gcur;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetDevice_v2(CUdevice *d, CUcontext c) {
  *d = c ? c->device : gcur;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetApiVersion(CUcontext c, unsigned int *v) {
  (void)c;
  *v = 3020;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetFlags(unsigned int *f) {
  *f = G.primary.flags;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxSetFlags(unsigned int f) {
  G.primary.flags = f;
  return CUDA_SUCCESS;
}
static size_t limits[16] = {[CU_LIMIT_STACK_SIZE] = 1024, [CU_LIMIT_PRINTF_FIFO_SIZE] = 1 << 20,
                            [CU_LIMIT_MALLOC_HEAP_SIZE] = 8 << 20,
                            [CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH] = 2,
                            [CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT] = 2048,
                            [CU_LIMIT_MAX_L2_FETCH_GRANULARITY] = 32};
CUresult CUDAAPI cuCtxSetLimit(CUlimit l, size_t v) {
  if ((unsigned)l < sizeof limits / sizeof limits[0]) limits[l] = v;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetLimit(size_t *v, CUlimit l) {
  *v = (unsigned)l < sizeof limits / sizeof limits[0] ? limits[l] : 0;
  queryTrace("cuCtxGetLimit(%d) = %zu", (int)l, *v);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *c) {
  *c = CU_FUNC_CACHE_PREFER_NONE;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache c) { (void)c; return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *least, int *greatest) {
  if (least) *least = 0;
  if (greatest) *greatest = -5;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetId(CUcontext c, unsigned long long *id) {
  *id = 1 + (unsigned long long)(c ? c->device : gcur);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxResetPersistingL2Cache(void) { return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext c, unsigned int f) {
  (void)f;
  return c && c->device != gcur ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
}
CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext c) {
  return c && c->device != gcur ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
}

// ---- Device memory: first fit over the part's memory, allocations kept sorted by address ----

struct Alloc {
  CUdeviceptr base;
  size_t size;
  int graphOwned;  // address reserved until the last graph or executable releases it
  int reserved;  // a virtual address range (cuMemAddressReserve), not memory in use
  unsigned long long id;
  struct Alloc *next;
};
static unsigned long long allocIds;

static Alloc *findAlloc(CUdeviceptr p) {
  int g = gpuAt(p);
  if (g < 0) return NULL;
  for (Alloc *a = gpus[g].allocs; a && a->base <= p; a = a->next)
    if (p < a->base + a->size) return a;
  return NULL;
}

static CUresult allocRange(CUdeviceptr *p, size_t bytes, size_t align, int reserved) {
  if (bytes == 0) return CUDA_ERROR_INVALID_VALUE;
  if (align < ALLOC_ALIGN) align = ALLOC_ALIGN;
  size_t size = (bytes + ALLOC_ALIGN - 1) & ~(size_t)(ALLOC_ALIGN - 1);
  pthread_mutex_lock(&lock);
  CUdeviceptr at = gpuBase(gcur), end = gpuBase(gcur) + G.part->memBytes;
  Alloc **q = &G.allocs;
  for (; *q; q = &(*q)->next) {
    if (at + size <= (*q)->base) break;
    CUdeviceptr e = (*q)->base + (*q)->size;
    at = (e + align - 1) & ~(CUdeviceptr)(align - 1);
  }
  if (at + size > end) {
    pthread_mutex_unlock(&lock);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  Alloc *a = malloc(sizeof *a);
  *a = (Alloc){.base = at, .size = size, .reserved = reserved, .id = ++allocIds, .next = *q};
  *q = a;
  if (!reserved) {
    G.inUse += size;
    publish((int64_t)size);
  }
  *p = at;
  pthread_mutex_unlock(&lock);
  return CUDA_SUCCESS;
}

static CUresult freeRange(CUdeviceptr p, int reserved) {
  pthread_mutex_lock(&lock);
  int was = gpuEnter(gpuAt(p));
  for (Alloc **q = &G.allocs; *q; q = &(*q)->next)
    if ((*q)->base == p && (*q)->reserved == reserved) {
      Alloc *a = *q;
      if (a->graphOwned) {
        if (G.lean) {
          enterLean();
          resWrote(a->base, a->size, 1);
          G.lean = nvgpu_release(G.lean, a->base, a->size);
          resWrote(a->base, a->size, 0);
        }
        gpuLeave(was);
        pthread_mutex_unlock(&lock);
        return CUDA_SUCCESS;
      }
      *q = a->next;
      if (!reserved) {
        G.inUse -= a->size;
        publish(-(int64_t)a->size);
        // A launch failure consumed the Lean device. Graph ownership still has to unwind.
        if (G.lean) {
          enterLean();
          resWrote(a->base, a->size, 1);
          G.lean = nvgpu_release(G.lean, a->base, a->size);
          resWrote(a->base, a->size, 0);
        }
      }
      free(a);
      gpuLeave(was);
      pthread_mutex_unlock(&lock);
      return CUDA_SUCCESS;
    }
  gpuLeave(was);
  pthread_mutex_unlock(&lock);
  return p == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *p, size_t bytes) {
  NEED_INIT();
  return allocRange(p, bytes, ALLOC_ALIGN, 0);
}

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *p, size_t *pitch, size_t width, size_t height,
                                 unsigned int elem) {
  NEED_INIT();
  (void)elem;
  *pitch = (width + 511) & ~(size_t)511;
  return allocRange(p, *pitch * height, ALLOC_ALIGN, 0);
}

CUresult CUDAAPI cuMemFree(CUdeviceptr p) {
  NEED_INIT();
  return freeRange(p, 0);
}

CUresult CUDAAPI cuMemGetInfo(size_t *freeBytes, size_t *total) {
  NEED_INIT();
  *total = G.part->memBytes;
  *freeBytes = G.part->memBytes - G.inUse;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *base, size_t *size, CUdeviceptr p) {
  pthread_mutex_lock(&lock);
  Alloc *a = findAlloc(p);
  if (a) {
    if (base) *base = a->base;
    if (size) *size = a->size;
  }
  pthread_mutex_unlock(&lock);
  return a ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

static int isDevice(CUdeviceptr x) { return gpuAt(x) >= 0; }

// ---- Host memory: page-locked memory is ordinary host memory, registered so that copies and
// pointer queries know it ----

typedef struct Host {
  void *p;
  size_t size;
  int owned;  // allocated here (cuMemHostAlloc), not registered (cuMemHostRegister)
  unsigned flags;
  struct Host *next;
} Host;
static Host *hosts;

static Host *findHost(const void *p) {
  for (Host *h = hosts; h; h = h->next)
    if ((const char *)p >= (char *)h->p && (const char *)p < (char *)h->p + h->size) return h;
  return NULL;
}

static void addHost(void *p, size_t n, int owned, unsigned flags) {
  Host *h = malloc(sizeof *h);
  *h = (Host){p, n, owned, flags, NULL};
  pthread_mutex_lock(&lock);
  h->next = hosts;
  hosts = h;
  pthread_mutex_unlock(&lock);
}

static int dropHost(void *p, int owned) {
  pthread_mutex_lock(&lock);
  for (Host **q = &hosts; *q; q = &(*q)->next)
    if ((*q)->p == p && (*q)->owned == owned) {
      Host *h = *q;
      *q = h->next;
      free(h);
      pthread_mutex_unlock(&lock);
      return 1;
    }
  pthread_mutex_unlock(&lock);
  return 0;
}

static int pinned(const void *p) {
  pthread_mutex_lock(&lock);
  int r = findHost(p) != NULL;
  pthread_mutex_unlock(&lock);
  return r;
}

CUresult CUDAAPI cuMemHostAlloc(void **p, size_t n, unsigned int flags) {
  *p = aligned_alloc(4096, (n + 4095) & ~(size_t)4095);
  if (!*p) return CUDA_ERROR_OUT_OF_MEMORY;
  addHost(*p, n, 1, flags);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemAllocHost(void **p, size_t n) { return cuMemHostAlloc(p, n, 0); }
CUresult CUDAAPI cuMemFreeHost(void *p) {
  if (!dropHost(p, 1)) return CUDA_ERROR_INVALID_VALUE;
  free(p);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemHostRegister(void *p, size_t n, unsigned int flags) {
  if (pinned(p)) return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
  addHost(p, n, 0, flags);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemHostUnregister(void *p) {
  return dropHost(p, 0) ? CUDA_SUCCESS : CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
}
CUresult CUDAAPI cuMemHostGetFlags(unsigned int *flags, void *p) {
  pthread_mutex_lock(&lock);
  Host *h = findHost(p);
  if (h) *flags = h->flags;
  pthread_mutex_unlock(&lock);
  return h ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
// Unified addressing: a page-locked host buffer's device address is its host address. Kernels do
// not reach host memory on the emulated device (its global memory is the device's), so a kernel
// that dereferences it faults, naming the address.
CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *d, void *p, unsigned int f) {
  (void)f;
  if (!pinned(p)) return CUDA_ERROR_INVALID_VALUE;
  *d = (CUdeviceptr)p;
  return CUDA_SUCCESS;
}

// ---- Pointer attributes ----

static CUresult pointerAttribute(void *data, CUpointer_attribute a, CUdeviceptr p) {
  pthread_mutex_lock(&lock);
  Alloc *al = isDevice(p) ? findAlloc(p) : NULL;
  int onDevice = isDevice(p);
  Host *h = onDevice ? NULL : findHost((void *)p);
  pthread_mutex_unlock(&lock);
  if (!onDevice && !h) return CUDA_ERROR_INVALID_VALUE;
  switch (a) {
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
      *(unsigned int *)data = onDevice ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_CONTEXT:
      *(CUcontext *)data = onDevice ? &gpus[gpuAt(p)].primary : &G.primary;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: *(CUdeviceptr *)data = p; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
      *(void **)data = onDevice ? NULL : (void *)p;
      return onDevice ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: *(int *)data = onDevice ? gpuAt(p) : gcur; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_IS_MANAGED: *(unsigned int *)data = 0; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_MAPPED: *(unsigned int *)data = 1; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: *(unsigned int *)data = 1; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE: *(unsigned int *)data = 0; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE: *(unsigned int *)data = 0; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: *(unsigned long long *)data = 0; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
      *(unsigned int *)data = CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: *(CUmemoryPool *)data = NULL; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
      *(unsigned long long *)data = al ? al->id : 0;
      return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
      if (al) *(CUdeviceptr *)data = al->base;
      else if (h) *(CUdeviceptr *)data = (CUdeviceptr)h->p;
      return al || h ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
      if (al) *(size_t *)data = al->size;
      else if (h) *(size_t *)data = h->size;
      return al || h ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    default: return CUDA_ERROR_INVALID_VALUE;
  }
}

CUresult CUDAAPI cuPointerGetAttribute(void *data, CUpointer_attribute a, CUdeviceptr p) {
  NEED_INIT();
  return pointerAttribute(data, a, p);
}

// Several attributes at once; memory the driver does not know is reported as such (memory type
// 0, no context), not as an error, as the driver does.
CUresult CUDAAPI cuPointerGetAttributes(unsigned int n, CUpointer_attribute *as, void **data,
                                        CUdeviceptr p) {
  NEED_INIT();
  int known = isDevice(p) || pinned((void *)p);
  for (unsigned i = 0; i < n; i++) {
    if (known && pointerAttribute(data[i], as[i], p) == CUDA_SUCCESS) continue;
    switch (as[i]) {
      case CU_POINTER_ATTRIBUTE_CONTEXT: *(CUcontext *)data[i] = NULL; break;
      case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: *(unsigned int *)data[i] = 0; break;
      case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
      case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR: *(CUdeviceptr *)data[i] = 0; break;
      case CU_POINTER_ATTRIBUTE_HOST_POINTER: *(void **)data[i] = NULL; break;
      case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: *(int *)data[i] = -2; break;
      case CU_POINTER_ATTRIBUTE_BUFFER_ID:
      case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: *(unsigned long long *)data[i] = 0; break;
      case CU_POINTER_ATTRIBUTE_RANGE_SIZE: *(size_t *)data[i] = 0; break;
      case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: *(CUmemoryPool *)data[i] = NULL; break;
      default: *(unsigned int *)data[i] = 0;
    }
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuPointerSetAttribute(const void *v, CUpointer_attribute a, CUdeviceptr p) {
  (void)v; (void)a; (void)p;
  return CUDA_SUCCESS;  // SYNC_MEMOPS: every copy is synchronous with respect to memory already
}

// ---- Virtual memory management: an address range, physical handles mapped into it. The device's
// memory is one store, so mapping is bookkeeping: a range is usable once reserved ----

typedef struct Phys {
  size_t size;
} Phys;

CUresult CUDAAPI cuMemGetAllocationGranularity(size_t *g, const CUmemAllocationProp *prop,
                                               CUmemAllocationGranularity_flags o) {
  (void)prop; (void)o;
  *g = 2 << 20;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr *p, size_t size, size_t align, CUdeviceptr addr,
                                     unsigned long long flags) {
  NEED_INIT();
  (void)addr; (void)flags;
  return allocRange(p, size, align ? align : (2 << 20), 1);
}
CUresult CUDAAPI cuMemAddressFree(CUdeviceptr p, size_t size) {
  (void)size;
  return freeRange(p, 1);
}
CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *h, size_t size,
                             const CUmemAllocationProp *prop, unsigned long long flags) {
  NEED_INIT();
  (void)prop; (void)flags;
  Phys *ph = malloc(sizeof *ph);
  ph->size = size;
  pthread_mutex_lock(&lock);
  G.inUse += size;
  publish((int64_t)size);
  pthread_mutex_unlock(&lock);
  *h = (CUmemGenericAllocationHandle)(uintptr_t)ph;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle h) {
  Phys *ph = (Phys *)(uintptr_t)h;
  pthread_mutex_lock(&lock);
  G.inUse -= ph->size;
  publish(-(int64_t)ph->size);
  pthread_mutex_unlock(&lock);
  free(ph);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemMap(CUdeviceptr p, size_t size, size_t offset, CUmemGenericAllocationHandle h,
                          unsigned long long flags) {
  (void)offset; (void)h; (void)flags;
  return isDevice(p) && isDevice(p + size - 1) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI cuMemUnmap(CUdeviceptr p, size_t size) {
  pthread_mutex_lock(&lock);
  int was = gpuEnter(gpuAt(p));
  enterLean();
  resWrote(p, size, 1);
  G.lean = nvgpu_release(G.lean, p, size);
  resWrote(p, size, 0);
  gpuLeave(was);
  pthread_mutex_unlock(&lock);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemSetAccess(CUdeviceptr p, size_t size, const CUmemAccessDesc *d, size_t n) {
  (void)p; (void)size; (void)d; (void)n;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemGetAccess(unsigned long long *flags, const CUmemLocation *l, CUdeviceptr p) {
  (void)l; (void)p;
  *flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *h, void *addr) {
  (void)h; (void)addr;
  note("cuMemRetainAllocationHandle: mappings are not kept");
  return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop,
                                                        CUmemGenericAllocationHandle h) {
  (void)h;
  memset(prop, 0, sizeof *prop);
  prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id = 0;
  return CUDA_SUCCESS;
}

// ---- Memory pools: the device's one memory, allocated from as cuMemAlloc does ----

struct CUmemPoolHandle_st {
  cuuint64_t threshold;
};
static struct CUmemPoolHandle_st defaultPool;

CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool *p, CUdevice d) {
  (void)d;
  *p = &defaultPool;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool *p, CUdevice d) { return cuDeviceGetDefaultMemPool(p, d); }
CUresult CUDAAPI cuDeviceSetMemPool(CUdevice d, CUmemoryPool p) {
  (void)d; (void)p;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool *p, const CUmemPoolProps *props) {
  (void)props;
  *p = calloc(1, sizeof(struct CUmemPoolHandle_st));
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool p) {
  if (p != &defaultPool) free(p);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool p, CUmemPool_attribute a, void *v) {
  if (a == CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) p->threshold = *(cuuint64_t *)v;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool p, CUmemPool_attribute a, void *v) {
  switch (a) {
    case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: *(cuuint64_t *)v = p->threshold; break;
    case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT:
    case CU_MEMPOOL_ATTR_USED_MEM_CURRENT: *(cuuint64_t *)v = G.inUse; break;
    case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:
    case CU_MEMPOOL_ATTR_USED_MEM_HIGH: *(cuuint64_t *)v = G.inUse; break;
    default: *(int *)v = 0;
  }
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool p, size_t keep) {
  (void)p; (void)keep;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool p, const CUmemAccessDesc *d, size_t n) {
  (void)p; (void)d; (void)n;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags *f, CUmemoryPool p, CUmemLocation *l) {
  (void)p; (void)l;
  *f = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return CUDA_SUCCESS;
}

#include "module.h"
#include "tex.h"
#include "stream.h"
#include "graph.h"
#include "accel.h"
#include "tmap.h"

// ---- Errors ----

static const struct {
  CUresult e;
  const char *name, *text;
} errors[] = {
  {CUDA_SUCCESS, "CUDA_SUCCESS", "no error"},
  {CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument"},
  {CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory"},
  {CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED", "initialization error"},
  {CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected"},
  {CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal"},
  {CUDA_ERROR_INVALID_IMAGE, "CUDA_ERROR_INVALID_IMAGE", "device kernel image is invalid"},
  {CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid device context"},
  {CUDA_ERROR_NO_BINARY_FOR_GPU, "CUDA_ERROR_NO_BINARY_FOR_GPU", "no kernel image is available for execution on the device"},
  {CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE", "invalid resource handle"},
  {CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found"},
  {CUDA_ERROR_NOT_READY, "CUDA_ERROR_NOT_READY", "device not ready"},
  {CUDA_ERROR_ILLEGAL_ADDRESS, "CUDA_ERROR_ILLEGAL_ADDRESS", "an illegal memory access was encountered"},
  {CUDA_ERROR_LAUNCH_FAILED, "CUDA_ERROR_LAUNCH_FAILED", "unspecified launch failure"},
  {CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", "too many resources requested for launch"},
  {CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported by the emulated device"},
  {CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED", "peer access is not supported between these two devices"},
  {CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED", "peer access has not been enabled"},
  {CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED", "operation not permitted when stream is capturing"},
  {CUDA_ERROR_STREAM_CAPTURE_UNMATCHED, "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED", "capture must end on its origin stream"},
  {CUDA_ERROR_STREAM_CAPTURE_UNJOINED, "CUDA_ERROR_STREAM_CAPTURE_UNJOINED", "a captured branch has not rejoined the origin"},
  {CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED", "operation failed due to a previous error during capture"},
  {CUDA_ERROR_ILLEGAL_STATE, "CUDA_ERROR_ILLEGAL_STATE", "the operation is not permitted in this state"},
  {CUDA_ERROR_OPERATING_SYSTEM, "CUDA_ERROR_OPERATING_SYSTEM", "OS call failed or operation not supported on this OS"},
  {CUDA_ERROR_FILE_NOT_FOUND, "CUDA_ERROR_FILE_NOT_FOUND", "file not found"},
  {CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED", "part or all of the requested memory range is already mapped"},
  {CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED", "pointer does not correspond to a registered memory region"},
};

CUresult CUDAAPI cuGetErrorString(CUresult e, const char **s) {
  // The device's own message for the failures it describes (which kernel, which address).
  if (lastError[0] && (e == sticky || e == CUDA_ERROR_INVALID_IMAGE || e == CUDA_ERROR_NOT_FOUND ||
                       e == CUDA_ERROR_NOT_SUPPORTED || e == CUDA_ERROR_NO_BINARY_FOR_GPU)) {
    *s = lastError;
    return CUDA_SUCCESS;
  }
  for (unsigned i = 0; i < sizeof errors / sizeof errors[0]; i++)
    if (errors[i].e == e) {
      *s = errors[i].text;
      return CUDA_SUCCESS;
    }
  *s = "unknown error";
  return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI cuGetErrorName(CUresult e, const char **s) {
  for (unsigned i = 0; i < sizeof errors / sizeof errors[0]; i++)
    if (errors[i].e == e) {
      *s = errors[i].name;
      return CUDA_SUCCESS;
    }
  *s = "CUDA_ERROR_UNKNOWN";
  return CUDA_ERROR_INVALID_VALUE;
}

// ---- Entry points by name ----

// A driver function the device does not provide was called (stubs.c, generated by gen.py).
void nvgpu_unsupported(const char *name) { note("%s is not provided by the emulated device", name); }

#include "procs.h"

// The driver functions whose ABI changed between CUDA versions, where this driver's headers name
// the newer one: a runtime asking for a version before `below` gets the older form (a CUDA 12
// runtime under this CUDA 13 driver asks for the CUDA 12 ABIs). Entries of a name in increasing
// `below`.
// legacy.c's older ABIs, by their exported names (cuda.h's macros rename these names in C).
extern char v1GraphExecUpdate[] __asm__("cuGraphExecUpdate");
extern char v1GetProcAddress[] __asm__("cuGetProcAddress");
extern char v1GraphInstantiate[] __asm__("cuGraphInstantiate");
extern char v1GraphAddKernelNode[] __asm__("cuGraphAddKernelNode");
extern char v1GraphKernelNodeGetParams[] __asm__("cuGraphKernelNodeGetParams");
extern char v1GraphKernelNodeSetParams[] __asm__("cuGraphKernelNodeSetParams");
extern char v1GraphExecKernelNodeSetParams[] __asm__("cuGraphExecKernelNodeSetParams");

static const struct {
  const char *name;
  int below;
  void *fn;
} legacy[] = {
  // cudaTypedefs.h: cuGetProcAddress v11030 (four arguments) / v12000 (status added).
  {"cuGetProcAddress", 12000, (void *)v1GetProcAddress},
  // CUDA 12.0 replaced these (instantiation flags; kernel node parameters with the kernel).
  {"cuGraphExecUpdate", 12000, (void *)v1GraphExecUpdate},
  {"cuGraphInstantiate", 12000, (void *)v1GraphInstantiate},
  {"cuGraphAddKernelNode", 12000, (void *)v1GraphAddKernelNode},
  {"cuGraphKernelNodeGetParams", 12000, (void *)v1GraphKernelNodeGetParams},
  {"cuGraphKernelNodeSetParams", 12000, (void *)v1GraphKernelNodeSetParams},
  {"cuGraphExecKernelNodeSetParams", 12000, (void *)v1GraphExecKernelNodeSetParams},
  {"cuStreamGetCaptureInfo", 11030, (void *)legacyStreamGetCaptureInfo_v1},
  {"cuStreamGetCaptureInfo", 12030, (void *)legacyStreamGetCaptureInfo},
#if CUDA_VERSION >= 13000
  {"cuCtxCreate", 11040, (void *)legacyCtxCreate},
  {"cuCtxCreate", 12050, (void *)legacyCtxCreate_v3},
  {"cuStreamUpdateCaptureDependencies", 12030, (void *)legacyStreamUpdateCaptureDependencies},
  {"cuGraphAddNode", 12030, (void *)legacyGraphAddNode},
  {"cuGraphGetEdges", 12030, (void *)legacyGraphGetEdges},
  {"cuGraphNodeGetDependencies", 12030, (void *)legacyNodeGetDependencies},
  {"cuGraphNodeGetDependentNodes", 12030, (void *)legacyNodeGetDependentNodes},
  {"cuGraphAddDependencies", 12030, (void *)legacyAddDependencies},
  {"cuGraphRemoveDependencies", 12030, (void *)legacyRemoveDependencies},
#endif
  {"", 0, NULL},
};

static void *findLegacy(const char *name, int version) {
  for (unsigned i = 0; i < sizeof legacy / sizeof legacy[0]; i++)
    if (version < legacy[i].below && strcmp(legacy[i].name, name) == 0) return legacy[i].fn;
  return NULL;
}

#include "trace.h"

// Tracing (NVGPU_LEAN_TRACE): the calls in progress on this thread, innermost last.
static __thread struct TraceCall {
  long i;
  void *ret;
} traceStack[256];
static __thread int traceDepth;
static int tracing;  // 1: failing calls; 2: every call; -1: off
static unsigned char traceVoid[TRACED];

void *nvgpu_trace_enter(long i, void *ret) {
  if (traceDepth < 256) traceStack[traceDepth] = (struct TraceCall){i, ret};
  traceDepth++;
  return traceTarget[i];
}
void *nvgpu_trace_leave(int result) {
  traceDepth--;
  long i = traceStack[traceDepth].i;
  if (traceVoid[i]) {
    if (tracing > 1) fprintf(stderr, "nvgpu-lean: %*s%s returned\n", 2 * traceDepth, "", traceName[i]);
    return traceStack[traceDepth].ret;
  }
  if (result != CUDA_SUCCESS || tracing > 1) {
    const char *name = "";
    cuGetErrorName((CUresult)result, &name);
    int failed = result != CUDA_SUCCESS && lastError[0];
    fprintf(stderr, "nvgpu-lean: %*s%s -> %s%s%s\n", 2 * traceDepth, "", traceName[i], name,
            failed ? ": " : "", failed ? lastError : "");
  }
  return traceStack[traceDepth].ret;
}

// The entry point `p` as handed out: its tracing trampoline when tracing (one per function).
static void *traced(const char *symbol, void *p, int returnsVoid) {
  if (!p) return p;
  if (!tracing) {
    const char *t = getenv("NVGPU_LEAN_TRACE");
    tracing = t && t[0] == '2' ? 2 : t && t[0] ? 1 : -1;
  }
  if (tracing < 0) return p;
  pthread_mutex_lock(&lock);
  void *r = p;
  for (int i = 0; i < TRACED; i++)
    if (traceTarget[i] == p || !traceTarget[i]) {
      if (!traceTarget[i]) traceName[i] = strdup(symbol);
      traceTarget[i] = p;
      traceVoid[i] = returnsVoid;
      r = traceTramp[i];
      break;
    }
  pthread_mutex_unlock(&lock);
  return r;
}

#include "runtime_tables.h"

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int version, cuuint64_t flags,
                                  CUdriverProcAddressQueryResult *status) {
  if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;
  (void)flags;
  void *p = findLegacy(symbol, version);
  // cudaTypedefs.h: v12050 adds the green-context output to cuStreamGetCtx.
  if (!p && version >= 12050 && strcmp(symbol, "cuStreamGetCtx") == 0)
    p = (void *)cuStreamGetCtx_v2;
  if (!p) p = findProc(symbol);
  p = traced(symbol, p, 0);
  if (tracing > 1) fprintf(stderr, "nvgpu-lean: cuGetProcAddress(%s, %d, %llu)%s\n", symbol, version,
                           (unsigned long long)flags, p ? "" : ": none");
  if (status) *status = p ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  *pfn = p;
  // Driver Entry Point Access: an absent symbol is reported through pfn and symbolStatus;
  // it is a successful query, not a failed driver operation (CUDA 13.0 cuda.h).
  return CUDA_SUCCESS;
}

// The unversioned entry point, as older runtimes look it up.
#undef cuGetProcAddress
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int version, cuuint64_t flags) {
  CUresult r = cuGetProcAddress_v2(symbol, pfn, version, flags, NULL);
  // CUDA 11's four-argument API has no status output and reports missing symbols as an error
  // (Driver Entry Point Access, CUDA 11.8), unlike the v12000 API's successful null result.
  return r == CUDA_SUCCESS && !*pfn ? CUDA_ERROR_NOT_FOUND : r;
}
