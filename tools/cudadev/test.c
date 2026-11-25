// The driver API on the emulated device, as frameworks use it, each part checked on the host:
// attributes; a PTX module (compiled by the shim through ptxas) with a `__device__` counter and a
// `__constant__` scale reached through cuModuleGetGlobal; launches; two streams ordered by an
// event, with asynchronous copies, sets, a host function and stream-ordered allocation; a graph
// built node by node and one captured from a stream, each launched more than once; a launch with
// a cluster shape (parts with clusters); tensor maps encoded and refused; occupancy queries; and
// the ordering of event recordings, destroyed executables and streams, and value waits.
#include <cuda.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>

static const char *ptx =
  ".version 8.0\n.target sm_80\n.address_size 64\n"
  ".global .align 4 .u32 counter;\n"
  ".const .align 4 .f32 scale = 0f3F800000;\n"
  ".visible .entry axpy(.param .u64 px_, .param .u64 py_, .param .f32 a_, .param .u32 n_) {\n"
  "  .reg .pred p; .reg .b32 i, n, t; .reg .f32 a, s, xv, yv; .reg .b64 px, py, o;\n"
  "  ld.param.u64 px, [px_]; ld.param.u64 py, [py_]; ld.param.f32 a, [a_]; ld.param.u32 n, [n_];\n"
  "  ld.const.f32 s, [scale]; mul.rn.f32 a, a, s;\n"
  "  mov.u32 i, %ctaid.x; mov.u32 t, %ntid.x; mul.lo.u32 i, i, t; mov.u32 t, %tid.x; add.u32 i, i, t;\n"
  "  setp.ge.u32 p, i, n; @p bra done;\n"
  "  mul.wide.u32 o, i, 4; add.s64 px, px, o; add.s64 py, py, o;\n"
  "  ld.global.f32 xv, [px]; ld.global.f32 yv, [py]; fma.rn.f32 yv, a, xv, yv; st.global.f32 [py], yv;\n"
  "done:\n  ret;\n}\n"
  ".visible .entry bump() {\n"
  "  .reg .b32 r;\n  atom.global.add.u32 r, [counter], 1;\n  ret;\n}\n";

static int failures;

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s; cuGetErrorString(r_, &s); \
  fprintf(stderr, "%s:%d %s: %d %s\n", __FILE__, __LINE__, #x, r_, s); exit(1); } } while (0)

static void check(const char *what, int ok) {
  printf("%s: %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) failures++;
}

static unsigned hostCalls;
static void CUDA_CB onHost(void *p) { (void)p; __atomic_add_fetch(&hostCalls, 1, __ATOMIC_SEQ_CST); }

static unsigned readCounter(CUdeviceptr c) {
  unsigned v;
  CK(cuMemcpyDtoH(&v, c, 4));
  return v;
}

static unsigned sequence[4], sequenceCount;
static void CUDA_CB recordNumber(void *p) {
  unsigned i = __atomic_fetch_add(&sequenceCount, 1, __ATOMIC_SEQ_CST);
  if (i < 4) sequence[i] = *(unsigned *)p;
}
static void *recordLegacy(void *p) { CK(cuEventRecord((CUevent)p, NULL)); return NULL; }

// Host gates hold work without relying on the relative speeds of the stream workers.
typedef struct Gate {
  pthread_mutex_t lock;
  pthread_cond_t cv;
  int entered, open, called;
  CUresult status;
} Gate;
static void gateInit(Gate *g) {
  memset(g, 0, sizeof *g);
  pthread_mutex_init(&g->lock, NULL);
  pthread_cond_init(&g->cv, NULL);
}
static void CUDA_CB gateHost(void *p) {
  Gate *g = p;
  pthread_mutex_lock(&g->lock);
  g->entered = 1;
  pthread_cond_broadcast(&g->cv);
  while (!g->open) pthread_cond_wait(&g->cv, &g->lock);
  pthread_mutex_unlock(&g->lock);
}
static void gateWait(Gate *g) {
  pthread_mutex_lock(&g->lock);
  while (!g->entered) pthread_cond_wait(&g->cv, &g->lock);
  pthread_mutex_unlock(&g->lock);
}
static void gateOpen(Gate *g) {
  pthread_mutex_lock(&g->lock);
  g->open = 1;
  pthread_cond_broadcast(&g->cv);
  pthread_mutex_unlock(&g->lock);
}
static void gateDestroy(Gate *g) {
  pthread_cond_destroy(&g->cv);
  pthread_mutex_destroy(&g->lock);
}
static void CUDA_CB failureCallback(CUstream s, CUresult status, void *p) {
  (void)s;
  Gate *g = p;
  pthread_mutex_lock(&g->lock);
  g->status = status;
  g->called++;
  pthread_cond_broadcast(&g->cv);
  pthread_mutex_unlock(&g->lock);
}
static void failureDelivery(void) {
  Gate gate;
  gateInit(&gate);
  CUstream s;
  CUevent e;
  CK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
  CK(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING));
  CK(cuLaunchHostFunc(s, gateHost, &gate));
  gateWait(&gate);
  CUmodule fm;
  CUfunction fault;
  CK(cuModuleLoadData(&fm,
    ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry fault() { trap; ret; }\n"));
  CK(cuModuleGetFunction(&fault, fm, "fault"));
  CUgraph graph;
  CUgraphExec exec;
  CUdeviceptr allocation;
  CK(cuStreamBeginCapture(s, CU_STREAM_CAPTURE_MODE_RELAXED));
  CK(cuMemAllocAsync(&allocation, 4096, s));
  CK(cuLaunchKernel(fault, 1, 1, 1, 32, 1, 1, 0, s, NULL, NULL));
  CK(cuMemFreeAsync(allocation, s));
  CK(cuStreamEndCapture(s, &graph));
  CK(cuGraphInstantiateWithFlags(&exec, graph, 0));
  CK(cuGraphDestroy(graph));
  CK(cuGraphLaunch(exec, s));
  CK(cuGraphExecDestroy(exec));
  unsigned before = hostCalls;
  CK(cuLaunchHostFunc(s, onHost, NULL));
  CK(cuEventRecord(e, s));
  CK(cuStreamAddCallback(s, failureCallback, &gate, 0));
  gateOpen(&gate);
  CUresult failure = cuEventSynchronize(e);
  check("failed recording wakes its waiter", failure == CUDA_ERROR_LAUNCH_FAILED);
  pthread_mutex_lock(&gate.lock);
  while (!gate.called) pthread_cond_wait(&gate.cv, &gate.lock);
  check("callback receives the queued device failure once", gate.called == 1 && gate.status == failure);
  pthread_mutex_unlock(&gate.lock);
  check("stream synchronization reports the launch error", cuStreamSynchronize(s) == failure);
  check("plain host function is skipped after device failure", hostCalls == before);
  CK(cuEventDestroy(e));
  CK(cuStreamDestroy(s));
  gateDestroy(&gate);
}

static void graphSubmissionChecks(void) {
  CUstream a, b;
  CUevent event;
  CUgraph g;
  CUgraphExec exec;
  CUgraphNode host, record;
  CK(cuStreamCreate(&a, CU_STREAM_NON_BLOCKING));
  CK(cuStreamCreate(&b, CU_STREAM_NON_BLOCKING));
  CK(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  CK(cuGraphCreate(&g, 0));
  unsigned one = 1, two = 2, three = 3;
  CUDA_HOST_NODE_PARAMS hp = {recordNumber, &one};
  CK(cuGraphAddHostNode(&host, g, NULL, 0, &hp));
  CK(cuGraphAddEventRecordNode(&record, g, &host, 1, event));
  CK(cuGraphInstantiateWithFlags(&exec, g, 0));
  Gate gate;
  gateInit(&gate);
  CK(cuLaunchHostFunc(a, gateHost, &gate));
  gateWait(&gate);
  sequenceCount = 0;
  CK(cuGraphLaunch(exec, a));
  check("graph recording is pending at launch submission", cuEventQuery(event) == CUDA_ERROR_NOT_READY);
  hp.userData = &two;
  CK(cuGraphExecHostNodeSetParams(exec, host, &hp));
  CK(cuGraphLaunch(exec, b));
  // b's launch must follow a's launch even though a is held behind a host gate.
  check("same executable on another stream remains pending", cuStreamQuery(b) == CUDA_ERROR_NOT_READY);
  hp.userData = &three;
  CK(cuGraphHostNodeSetParams(host, &hp));
  CK(cuGraphHostNodeGetParams(host, &hp));
  CUgraphExecUpdateResultInfo info;
  CK(cuGraphExecUpdate(exec, g, &info));
  CK(cuGraphLaunch(exec, a));
  CK(cuGraphExecDestroy(exec));
  CK(cuGraphDestroy(g));
  gateOpen(&gate);
  CK(cuStreamSynchronize(a));
  CK(cuStreamSynchronize(b));
  check("queued launches keep parameters and submission order", sequenceCount == 3 &&
        sequence[0] == 1 && sequence[1] == 2 && sequence[2] == 3);
  CK(cuEventSynchronize(event));
  gateDestroy(&gate);

  // A capture's external record remains executable, including when captured as a child graph.
  CK(cuStreamBeginCapture(a, CU_STREAM_CAPTURE_MODE_RELAXED));
  CK(cuEventRecordWithFlags(event, a, CU_EVENT_RECORD_EXTERNAL));
  CK(cuStreamEndCapture(a, &g));
  CK(cuGraphInstantiateWithFlags(&exec, g, 0));
  CK(cuGraphDestroy(g));
  CK(cuStreamBeginCapture(b, CU_STREAM_CAPTURE_MODE_RELAXED));
  CK(cuGraphLaunch(exec, b));
  CK(cuStreamEndCapture(b, &g));
  CK(cuGraphExecDestroy(exec));
  CK(cuGraphInstantiateWithFlags(&exec, g, 0));
  gateInit(&gate);
  CK(cuLaunchHostFunc(b, gateHost, &gate));
  gateWait(&gate);
  CK(cuGraphLaunch(exec, b));
  check("captured external record publishes a new recording", cuEventQuery(event) == CUDA_ERROR_NOT_READY);
  gateOpen(&gate);
  CK(cuStreamSynchronize(b));
  CK(cuEventSynchronize(event));
  gateDestroy(&gate);
  CK(cuGraphExecDestroy(exec));
  CK(cuGraphDestroy(g));

  // A branch can only end on its origin, and all branch work must join back to that origin.
  CK(cuStreamBeginCapture(a, CU_STREAM_CAPTURE_MODE_RELAXED));
  CK(cuLaunchHostFunc(a, onHost, NULL));
  CK(cuEventRecord(event, a));
  CK(cuStreamWaitEvent(b, event, 0));
  CUgraph wrong;
  check("capture cannot end on a joined stream",
        cuStreamEndCapture(b, &wrong) == CUDA_ERROR_STREAM_CAPTURE_UNMATCHED);
  CK(cuLaunchHostFunc(b, onHost, NULL));
  check("unjoined branch is reported", cuStreamEndCapture(a, &wrong) == CUDA_ERROR_STREAM_CAPTURE_UNJOINED);

  CK(cuStreamWaitEvent(b, event, 0));
  CUstreamCaptureStatus state;
  CK(cuStreamIsCapturing(b, &state));
  check("old captured event cannot restart capture", state == CU_STREAM_CAPTURE_STATUS_NONE);

  // Allocations belong to every surviving executable, then return to the allocator.
  size_t before, after, total;
  CK(cuMemGetInfo(&before, &total));
  CK(cuStreamBeginCapture(a, CU_STREAM_CAPTURE_MODE_RELAXED));
  CUdeviceptr allocation;
  CK(cuMemAllocAsync(&allocation, 4096, a));
  CK(cuMemsetD8Async(allocation, 0x5a, 4096, a));
  CK(cuMemFreeAsync(allocation, a));
  CK(cuStreamEndCapture(a, &g));
  CK(cuGraphInstantiateWithFlags(&exec, g, 0));
  CK(cuGraphDestroy(g));
  for (int i = 0; i < 2; i++) { CK(cuGraphLaunch(exec, a)); CK(cuStreamSynchronize(a)); }
  CK(cuGraphExecDestroy(exec));
  CK(cuMemGetInfo(&after, &total));
  check("captured allocation released with last executable", after == before);

  // Conditional handles also own device storage and survive destruction of the source graph.
  CK(cuMemGetInfo(&before, &total));
  CK(cuGraphCreate(&g, 0));
  CUgraphConditionalHandle condition;
  CK(cuGraphConditionalHandleCreate(&condition, g, NULL, 1, CU_GRAPH_COND_ASSIGN_DEFAULT));
  CK(cuGraphInstantiateWithFlags(&exec, g, 0));
  CK(cuGraphDestroy(g));
  CK(cuMemGetInfo(&after, &total));
  check("conditional storage retained by executable", after < before);
  CK(cuGraphLaunch(exec, a));
  CK(cuStreamSynchronize(a));
  CK(cuGraphExecDestroy(exec));
  CK(cuMemGetInfo(&after, &total));
  check("conditional storage released with last executable", after == before);

  // A while node whose body's kernel counts and sets the condition from the device
  // (`cudaGraphSetConditional`): it runs while the count is below five.
  {
    static const char *condPtx =
      ".version 8.0\n.target sm_80\n.address_size 64\n"
      ".extern .func cudaGraphSetConditional (.param .b64 h, .param .b32 v);\n"
      ".visible .entry step(.param .u64 h_, .param .u64 c_) {\n"
      "  .reg .b64 h, c; .reg .b32 n, more; .reg .pred p;\n"
      "  ld.param.u64 h, [h_]; ld.param.u64 c, [c_];\n"
      "  ld.global.u32 n, [c]; add.u32 n, n, 1; st.global.u32 [c], n;\n"
      "  setp.lt.u32 p, n, 5; selp.u32 more, 1, 0, p;\n"
      "  { .param .b64 a0; .param .b32 a1; st.param.b64 [a0], h; st.param.b32 [a1], more;\n"
      "    call.uni cudaGraphSetConditional, (a0, a1); }\n"
      "  ret;\n}\n";
    CUmodule cm;
    CUfunction step;
    CUresult lr = cuModuleLoadData(&cm, condPtx);
    int ok = lr == CUDA_SUCCESS && cuModuleGetFunction(&step, cm, "step") == CUDA_SUCCESS;
    if (ok) {
      CUdeviceptr count;
      CK(cuMemAlloc(&count, 4));
      CK(cuMemsetD32(count, 0, 1));
      CK(cuGraphCreate(&g, 0));
      CUgraphConditionalHandle h;
      CK(cuGraphConditionalHandleCreate(&h, g, NULL, 1, CU_GRAPH_COND_ASSIGN_DEFAULT));
      CUgraphNodeParams np;
      memset(&np, 0, sizeof np);
      np.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
      np.conditional.handle = h;
      np.conditional.type = CU_GRAPH_COND_TYPE_WHILE;
      np.conditional.size = 1;
      CUgraphNode cn;
      CK(cuGraphAddNode(&cn, g, NULL, NULL, 0, &np));
      CUgraph body = np.conditional.phGraph_out[0];
      void *kargs[] = {&h, &count};
      CUDA_KERNEL_NODE_PARAMS kp;
      memset(&kp, 0, sizeof kp);
      kp.func = step;
      kp.gridDimX = kp.gridDimY = kp.gridDimZ = 1;
      kp.blockDimX = kp.blockDimY = kp.blockDimZ = 1;
      kp.kernelParams = kargs;
      CUgraphNode kn;
      CK(cuGraphAddKernelNode(&kn, body, NULL, 0, &kp));
      CK(cuGraphInstantiateWithFlags(&exec, g, 0));
      CK(cuGraphLaunch(exec, a));
      CK(cuStreamSynchronize(a));
      unsigned n = 0;
      CK(cuMemcpyDtoH(&n, count, 4));
      ok = n == 5;
      CK(cuGraphExecDestroy(exec));
      CK(cuGraphDestroy(g));
      CK(cuMemFree(count));
      CK(cuModuleUnload(cm));
    } else {
      printf("conditional kernel did not load: %d\n", (int)lr);
    }
    check("a while node runs until its body's kernel clears the condition from the device", ok);
  }

  // The pre-12 ABI writes separate outputs, including on a failed topology update.
  CK(cuGraphCreate(&g, 0));
  CK(cuGraphInstantiateWithFlags(&exec, g, 0));
  CK(cuGraphAddEmptyNode(&host, g, NULL, 0));
  CUresult (*update1)(CUgraphExec, CUgraph, CUgraphNode *, CUgraphExecUpdateResult *);
  CK(cuGetProcAddress("cuGraphExecUpdate", (void **)&update1, 11080, 0, NULL));
  struct { CUgraphNode node; unsigned long long guard[2]; } old = {0, {0x12345678, 0xabcdef}};
  CUgraphExecUpdateResult result = CU_GRAPH_EXEC_UPDATE_SUCCESS;
  check("old update ABI returns topology failure", update1(exec, g, &old.node, &result) == CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE &&
        result == CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED);
  check("old update ABI preserves adjacent outputs", old.guard[0] == 0x12345678 && old.guard[1] == 0xabcdef);
  CK(cuGraphExecDestroy(exec));
  CK(cuGraphDestroy(g));
  CK(cuEventDestroy(event));
  CK(cuStreamDestroy(a));
  CK(cuStreamDestroy(b));
}

static void legacySubmissionChecks(void) {
  CUstream a, b;
  CUevent e;
  Gate gate;
  gateInit(&gate);
  CK(cuStreamCreate(&a, 0));
  CK(cuStreamCreate(&b, 0));
  CK(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING));
  CK(cuLaunchHostFunc(a, gateHost, &gate));
  gateWait(&gate);
  pthread_t thread;
  pthread_create(&thread, NULL, recordLegacy, e);
  while (cuEventQuery(e) == CUDA_SUCCESS) sched_yield();
  // This wait was submitted after the legacy recording; it must not hold that recording up.
  CK(cuStreamWaitEvent(b, e, 0));
  unsigned before = hostCalls;
  CK(cuLaunchHostFunc(b, onHost, NULL));
  gateOpen(&gate);
  pthread_join(thread, NULL);
  CK(cuCtxSynchronize());
  check("legacy ordering uses only prior submissions", hostCalls == before + 1);
  CK(cuEventDestroy(e));
  CK(cuStreamDestroy(a));
  CK(cuStreamDestroy(b));
  gateDestroy(&gate);
}

// A kernel on device 0 reading and writing device 1's memory through device 1's pointers
// (`axpy`, y = 2x + y): x on device 1, y on device 0, then y on device 1 (its values copied back
// from device 1).
static void peerChecks(void) {
  int n;
  CK(cuDeviceGetCount(&n));
  check("a mesh of two devices", n >= 2);
  if (n < 2) return;
  CUdevice d0, d1;
  CK(cuDeviceGet(&d0, 0));
  CK(cuDeviceGet(&d1, 1));
  int can;
  CK(cuDeviceCanAccessPeer(&can, d0, d1));
  CUcontext c0, c1;
  CK(cuDevicePrimaryCtxRetain(&c0, d0));
  CK(cuDevicePrimaryCtxRetain(&c1, d1));
  enum { N = 1000 };
  float x[N], y[N], z[N];
  for (int i = 0; i < N; i++) { x[i] = (float)i; y[i] = 1.0f; }
  CUdeviceptr dx, dy, dy1;
  CK(cuCtxSetCurrent(c1));
  CK(cuMemAlloc(&dx, sizeof x));
  CK(cuMemAlloc(&dy1, sizeof y));
  CK(cuMemcpyHtoD(dx, x, sizeof x));
  CK(cuMemcpyHtoD(dy1, y, sizeof y));
  CK(cuCtxSetCurrent(c0));
  CK(cuCtxEnablePeerAccess(c1, 0));
  CK(cuMemAlloc(&dy, sizeof y));
  CK(cuMemcpyHtoD(dy, y, sizeof y));
  CUmodule m;
  CK(cuModuleLoadData(&m, ptx));
  CUfunction axpy;
  CK(cuModuleGetFunction(&axpy, m, "axpy"));
  float a = 2.0f;
  unsigned un = N;
  void *args[] = {&dx, &dy, &a, &un};
  CK(cuLaunchKernel(axpy, (N + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(z, dy, sizeof z));
  int ok = can == 1;
  for (int i = 0; i < N; i++) ok = ok && z[i] == 2.0f * (float)i + 1.0f;
  check("a kernel reads a peer's memory", ok);
  void *args1[] = {&dx, &dy1, &a, &un};
  CK(cuLaunchKernel(axpy, (N + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args1, NULL));
  CK(cuCtxSynchronize());
  CK(cuCtxSetCurrent(c1));
  CK(cuMemcpyDtoH(z, dy1, sizeof z));
  ok = 1;
  for (int i = 0; i < N; i++) ok = ok && z[i] == 2.0f * (float)i + 1.0f;
  check("a kernel writes a peer's memory", ok);
  CK(cuMemFree(dx));
  CK(cuMemFree(dy1));
  CK(cuCtxSetCurrent(c0));
  CK(cuMemFree(dy));
  CK(cuModuleUnload(m));
}

// A CUDA array read through a texture object and written through a surface object: an 8 × 4
// array of floats (row y, column x holds 10y + x) copied in with cuMemcpy2D; each thread (x, y)
// fetches texel (x − 1, y), clamped at the left edge, and writes x + 100y to a second array through
// a surface, which comes back with cuMemcpy2D.
static void textureChecks(void) {
  enum { W = 8, H = 4 };
  static const char *texPtx =
    ".version 8.0\n.target sm_80\n.address_size 64\n"
    ".visible .entry texk(.param .u64 tex_, .param .u64 surf_, .param .u64 out_) {\n"
    "  .reg .b64 t, s, o, a; .reg .f32 f0, f1, f2, f3; .reg .b32 x, y, xm, xb, v, i;\n"
    "  ld.param.u64 t, [tex_]; ld.param.u64 s, [surf_]; ld.param.u64 o, [out_];\n"
    "  mov.u32 x, %tid.x; mov.u32 y, %tid.y; sub.s32 xm, x, 1;\n"
    "  tex.2d.v4.f32.s32 {f0, f1, f2, f3}, [t, {xm, y}];\n"
    "  mad.lo.u32 i, y, 8, x; mul.wide.u32 a, i, 4; add.s64 a, o, a; st.global.f32 [a], f0;\n"
    "  mad.lo.u32 v, y, 100, x; shl.b32 xb, x, 2;\n"
    "  sust.b.2d.b32.trap [s, {xb, y}], {v};\n"
    "  ret;\n}\n";
  float in[H][W], out[H * W];
  unsigned back[H][W];
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) in[y][x] = (float)(10 * y + x);
  CUarray tarr, sarr;
  CUDA_ARRAY3D_DESCRIPTOR ad = {W, H, 0, CU_AD_FORMAT_FLOAT, 1, 0};
  CK(cuArray3DCreate(&tarr, &ad));
  ad.Flags = CUDA_ARRAY3D_SURFACE_LDST;
  ad.Format = CU_AD_FORMAT_UNSIGNED_INT32;
  CK(cuArray3DCreate(&sarr, &ad));
  CUDA_MEMCPY2D c;
  memset(&c, 0, sizeof c);
  c.srcMemoryType = CU_MEMORYTYPE_HOST;
  c.srcHost = in;
  c.srcPitch = sizeof in[0];
  c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  c.dstArray = tarr;
  c.WidthInBytes = sizeof in[0];
  c.Height = H;
  CK(cuMemcpy2D(&c));
  CUDA_RESOURCE_DESC rd;
  memset(&rd, 0, sizeof rd);
  rd.resType = CU_RESOURCE_TYPE_ARRAY;
  rd.res.array.hArray = tarr;
  CUDA_TEXTURE_DESC td;
  memset(&td, 0, sizeof td);
  td.addressMode[0] = td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  td.filterMode = CU_TR_FILTER_MODE_POINT;
  CUtexObject tex;
  CK(cuTexObjectCreate(&tex, &rd, &td, NULL));
  rd.res.array.hArray = sarr;
  CUsurfObject surf;
  CK(cuSurfObjectCreate(&surf, &rd));
  CUmodule m;
  CK(cuModuleLoadData(&m, texPtx));
  CUfunction k;
  CK(cuModuleGetFunction(&k, m, "texk"));
  CUdeviceptr dout;
  CK(cuMemAlloc(&dout, sizeof out));
  void *args[] = {&tex, &surf, &dout};
  CK(cuLaunchKernel(k, 1, 1, 1, W, H, 1, 0, NULL, args, NULL));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(out, dout, sizeof out));
  int ok = 1;
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) ok = ok && out[y * W + x] == in[y][x > 0 ? x - 1 : 0];
  check("a texture object fetches a CUDA array's texels, clamped", ok);

  // Linear filtering of a float array, as a GH200 filters (8 by 8 texels, (3,3) one and (4,3)
  // two): the far corner's weight the axes' 8-bit weights' product rounded to 1/256, the others
  // by difference, the sum rounded ties away.
  {
    static const char *linPtx =
      ".version 8.0\n.target sm_80\n.address_size 64\n"
      ".visible .entry link(.param .u64 tex_, .param .u64 xy_, .param .u64 out_) {\n"
      "  .reg .b64 t, p, o, a; .reg .b32 i; .reg .f32 x, y, r0, r1, r2, r3;\n"
      "  ld.param.u64 t, [tex_]; ld.param.u64 p, [xy_]; ld.param.u64 o, [out_];\n"
      "  mov.u32 i, %tid.x; mul.wide.u32 a, i, 8; add.s64 p, p, a; mul.wide.u32 a, i, 4; add.s64 o, o, a;\n"
      "  ld.global.f32 x, [p]; ld.global.f32 y, [p+4];\n"
      "  tex.2d.v4.f32.f32 {r0, r1, r2, r3}, [t, {x, y}];\n"
      "  st.global.f32 [o], r0;\n"
      "  ret;\n}\n";
    static const uint32_t cases[6][3] = {
      {0x40984bc8, 0x40459f10, 0x3f5e0000}, {0x406d0963, 0x408c3873, 0x3e100000},
      {0x408227bf, 0x40749eed, 0x3f888000}, {0x407d12c5, 0x40639b02, 0x3fb00000},
      {0x407abf87, 0x407f637c, 0x3f380000}, {0x40845e08, 0x408c2ea4, 0x3e4c0000}};
    CUDA_ARRAY_DESCRIPTOR ld8 = {8, 8, CU_AD_FORMAT_FLOAT, 1};
    CUarray la;
    CK(cuArrayCreate(&la, &ld8));
    float lt[64] = {0};
    lt[3 * 8 + 3] = 1.0f;
    lt[3 * 8 + 4] = 2.0f;
    CUDA_MEMCPY2D lc;
    memset(&lc, 0, sizeof lc);
    lc.srcMemoryType = CU_MEMORYTYPE_HOST; lc.srcHost = lt; lc.srcPitch = 32;
    lc.dstMemoryType = CU_MEMORYTYPE_ARRAY; lc.dstArray = la; lc.WidthInBytes = 32; lc.Height = 8;
    CK(cuMemcpy2D(&lc));
    CUDA_RESOURCE_DESC lr;
    memset(&lr, 0, sizeof lr);
    lr.resType = CU_RESOURCE_TYPE_ARRAY;
    lr.res.array.hArray = la;
    CUDA_TEXTURE_DESC ltd;
    memset(&ltd, 0, sizeof ltd);
    ltd.addressMode[0] = ltd.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    ltd.filterMode = CU_TR_FILTER_MODE_LINEAR;
    CUtexObject ltex;
    CK(cuTexObjectCreate(&ltex, &lr, &ltd, NULL));
    uint32_t xy[12], lout[6];
    for (int c = 0; c < 6; c++) { xy[2 * c] = cases[c][0]; xy[2 * c + 1] = cases[c][1]; }
    CUdeviceptr dxy, dlo;
    CK(cuMemAlloc(&dxy, sizeof xy));
    CK(cuMemAlloc(&dlo, sizeof lout));
    CK(cuMemcpyHtoD(dxy, xy, sizeof xy));
    CUmodule lm;
    CK(cuModuleLoadData(&lm, linPtx));
    CUfunction lk;
    CK(cuModuleGetFunction(&lk, lm, "link"));
    void *largs[] = {&ltex, &dxy, &dlo};
    CK(cuLaunchKernel(lk, 1, 1, 1, 6, 1, 1, 0, NULL, largs, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(lout, dlo, sizeof lout));
    int lok = 1;
    for (int c = 0; c < 6; c++) {
      lok = lok && lout[c] == cases[c][2];
      if (lout[c] != cases[c][2]) printf("linear case %d: %08x, a GH200 %08x\n", c, lout[c], cases[c][2]);
    }
    check("linear filtering of a float array weighs texels as a GH200 does", lok);
    CK(cuTexObjectDestroy(ltex));
    CK(cuModuleUnload(lm));

    // In three dimensions (16 by 16 by 16 texels, zero but for four fetches' eight each, from a
    // GH200's run over random texels): along z, then x, then y, each part rounded to 1/256.
    static const char *lin3Ptx =
      ".version 8.0\n.target sm_80\n.address_size 64\n"
      ".visible .entry lin3(.param .u64 tex_, .param .u64 xyz_, .param .u64 out_) {\n"
      "  .reg .b64 t, p, o, a; .reg .b32 i; .reg .f32 x, y, z, r0, r1, r2, r3;\n"
      "  ld.param.u64 t, [tex_]; ld.param.u64 p, [xyz_]; ld.param.u64 o, [out_];\n"
      "  mov.u32 i, %tid.x; mul.wide.u32 a, i, 12; add.s64 p, p, a; mul.wide.u32 a, i, 4; add.s64 o, o, a;\n"
      "  ld.global.f32 x, [p]; ld.global.f32 y, [p+4]; ld.global.f32 z, [p+8];\n"
      "  tex.3d.v4.f32.f32 {r0, r1, r2, r3}, [t, {x, y, z, z}];\n"
      "  st.global.f32 [o], r0;\n"
      "  ret;\n}\n";
    static const uint32_t at3[4][4] = {
      {0x403ec51a, 0x40c98719, 0x415961c3, 0x4010581d}, {0x411c401b, 0x416c5f82, 0x416ddc7e, 0x3feaa946},
      {0x401f0700, 0x4105a833, 0x4007ba7a, 0x402169f6}, {0x40ba8ef2, 0x40997772, 0x413fc75d, 0x3fe2ef23}};
    static const uint32_t texels3[32][4] = {
      {2, 5, 13, 0x4047aa90}, {3, 5, 13, 0x3fa8cdce}, {2, 6, 13, 0x402acfc3}, {3, 6, 13, 0x3fe44d94},
      {2, 5, 14, 0x3f812659}, {3, 5, 14, 0x3f55ad72}, {2, 6, 14, 0x3feba520}, {3, 6, 14, 0x405d9fb0},
      {9, 14, 14, 0x3fb0872c}, {10, 14, 14, 0x40019528}, {9, 15, 14, 0x3fd759ec}, {10, 15, 14, 0x4019c970},
      {9, 14, 15, 0x3ffd0b3c}, {10, 14, 15, 0x4043c60d}, {9, 15, 15, 0x3fcc1fdb}, {10, 15, 15, 0x40314685},
      {1, 7, 1, 0x3fe6300e}, {2, 7, 1, 0x3fb1ca40}, {1, 8, 1, 0x3ff37f7e}, {2, 8, 1, 0x3fdb41b3},
      {1, 7, 2, 0x3fac210e}, {2, 7, 2, 0x401c2a1d}, {1, 8, 2, 0x3f215cbb}, {2, 8, 2, 0x404b9806},
      {5, 4, 11, 0x3fc40518}, {6, 4, 11, 0x3f0afa40}, {5, 5, 11, 0x3f635202}, {6, 5, 11, 0x3f9927be},
      {5, 4, 12, 0x402ae558}, {6, 4, 12, 0x3fb262d2}, {5, 5, 12, 0x4049193c}, {6, 5, 12, 0x40189fe8}};
    static uint32_t vol[16 * 16 * 16];
    for (int e = 0; e < 32; e++) vol[texels3[e][0] + 16 * texels3[e][1] + 256 * texels3[e][2]] = texels3[e][3];
    CUDA_ARRAY3D_DESCRIPTOR vd = {16, 16, 16, CU_AD_FORMAT_FLOAT, 1, 0};
    CUarray va;
    CK(cuArray3DCreate(&va, &vd));
    CUDA_MEMCPY3D vc;
    memset(&vc, 0, sizeof vc);
    vc.srcMemoryType = CU_MEMORYTYPE_HOST; vc.srcHost = vol; vc.srcPitch = 64; vc.srcHeight = 16;
    vc.dstMemoryType = CU_MEMORYTYPE_ARRAY; vc.dstArray = va;
    vc.WidthInBytes = 64; vc.Height = 16; vc.Depth = 16;
    CK(cuMemcpy3D(&vc));
    lr.res.array.hArray = va;
    ltd.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    CUtexObject vtex;
    CK(cuTexObjectCreate(&vtex, &lr, &ltd, NULL));
    uint32_t xyz[12], vout[4];
    for (int c = 0; c < 4; c++) for (int e = 0; e < 3; e++) xyz[3 * c + e] = at3[c][e];
    CUdeviceptr dxyz, dvo;
    CK(cuMemAlloc(&dxyz, sizeof xyz));
    CK(cuMemAlloc(&dvo, sizeof vout));
    CK(cuMemcpyHtoD(dxyz, xyz, sizeof xyz));
    CK(cuModuleLoadData(&lm, lin3Ptx));
    CK(cuModuleGetFunction(&lk, lm, "lin3"));
    void *vargs[] = {&vtex, &dxyz, &dvo};
    CK(cuLaunchKernel(lk, 1, 1, 1, 4, 1, 1, 0, NULL, vargs, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(vout, dvo, sizeof vout));
    int vok = 1;
    for (int c = 0; c < 4; c++) {
      vok = vok && vout[c] == at3[c][3];
      if (vout[c] != at3[c][3]) printf("3d linear case %d: %08x, a GH200 %08x\n", c, vout[c], at3[c][3]);
    }
    check("linear filtering of a 3D float array weighs texels as a GH200 does", vok);
    CK(cuTexObjectDestroy(vtex));
    CK(cuModuleUnload(lm));
    CK(cuMemFree(dxyz));
    CK(cuMemFree(dvo));
    CK(cuArrayDestroy(va));
    CK(cuMemFree(dxy));
    CK(cuMemFree(dlo));
    CK(cuArrayDestroy(la));
  }
  memset(&c, 0, sizeof c);
  c.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  c.srcArray = sarr;
  c.dstMemoryType = CU_MEMORYTYPE_HOST;
  c.dstHost = back;
  c.dstPitch = sizeof back[0];
  c.WidthInBytes = sizeof back[0];
  c.Height = H;
  CK(cuMemcpy2D(&c));
  ok = 1;
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) ok = ok && back[y][x] == (unsigned)(x + 100 * y);
  check("a surface object writes a CUDA array", ok);
  // A view of the float array as 32-bit unsigned integers fetches the texels' bits.
  static const char *viewPtx =
    ".version 8.0\n.target sm_80\n.address_size 64\n"
    ".visible .entry viewk(.param .u64 tex_, .param .u64 out_) {\n"
    "  .reg .b64 t, o, a; .reg .b32 x, y, i, r0, r1, r2, r3;\n"
    "  ld.param.u64 t, [tex_]; ld.param.u64 o, [out_];\n"
    "  mov.u32 x, %tid.x; mov.u32 y, %tid.y;\n"
    "  tex.2d.v4.u32.s32 {r0, r1, r2, r3}, [t, {x, y}];\n"
    "  mad.lo.u32 i, y, 8, x; mul.wide.u32 a, i, 4; add.s64 a, o, a; st.global.u32 [a], r0;\n"
    "  ret;\n}\n";
  CUDA_RESOURCE_VIEW_DESC vd;
  memset(&vd, 0, sizeof vd);
  vd.format = CU_RES_VIEW_FORMAT_UINT_1X32;
  rd.res.array.hArray = tarr;
  CUtexObject vtex;
  CK(cuTexObjectCreate(&vtex, &rd, &td, &vd));
  CUmodule vm;
  CK(cuModuleLoadData(&vm, viewPtx));
  CUfunction vk;
  CK(cuModuleGetFunction(&vk, vm, "viewk"));
  void *vargs[] = {&vtex, &dout};
  CK(cuLaunchKernel(vk, 1, 1, 1, W, H, 1, 0, NULL, vargs, NULL));
  CK(cuCtxSynchronize());
  unsigned bits[H * W];
  CK(cuMemcpyDtoH(bits, dout, sizeof bits));
  ok = 1;
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) ok = ok && !memcmp(&bits[y * W + x], &in[y][x], 4);
  vd.format = CU_RES_VIEW_FORMAT_UINT_1X16;
  CUtexObject bad;
  ok = ok && cuTexObjectCreate(&bad, &rd, &td, &vd) == CUDA_ERROR_INVALID_VALUE;
  check("a resource view reads a float array's texels as integers, and refuses another size", ok);
  CK(cuTexObjectDestroy(vtex));
  CK(cuModuleUnload(vm));

  // A mipmapped array of 64 by 64 floats, 7 levels, level k all 2^k: an explicit-LOD fetch at its
  // middle reads the nearest level, or blends the two about the LOD (a GH200 and an A100 both:
  // LOD 0.5 reads level 1, 1.25 reads 2.5, a bias of 0.3 adds 76/256, the LOD -0.2966 is -75/256
  // (truncated toward zero), clamps hold the LOD); a blend that ties rounds away from zero.
  {
    static const char *mipPtx =
      ".version 8.0\n.target sm_80\n.address_size 64\n"
      ".visible .entry mipk(.param .u64 tex_, .param .u64 out_, .param .u64 lods_) {\n"
      "  .reg .b64 t, o, l, a; .reg .b32 i; .reg .f32 x, lod, r0, r1, r2, r3;\n"
      "  ld.param.u64 t, [tex_]; ld.param.u64 o, [out_]; ld.param.u64 l, [lods_];\n"
      "  mov.u32 i, %tid.x; mul.wide.u32 a, i, 4; add.s64 l, l, a; add.s64 o, o, a;\n"
      "  ld.global.f32 lod, [l]; mov.f32 x, 0f3F000000;\n"
      "  tex.level.2d.v4.f32.f32 {r0, r1, r2, r3}, [t, {x, x}], lod;\n"
      "  st.global.f32 [o], r0;\n"
      "  ret;\n}\n";
    CUDA_ARRAY3D_DESCRIPTOR md = {64, 64, 0, CU_AD_FORMAT_FLOAT, 1, 0};
    CUmipmappedArray mip;
    CK(cuMipmappedArrayCreate(&mip, &md, 7));
    for (unsigned k = 0; k < 7; k++) {
      CUarray lv;
      CK(cuMipmappedArrayGetLevel(&lv, mip, k));
      unsigned w = 64 >> k;
      float lvl[64 * 64];
      for (unsigned e = 0; e < w * w; e++) lvl[e] = (float)(1u << k);
      CUDA_MEMCPY2D cp;
      memset(&cp, 0, sizeof cp);
      cp.srcMemoryType = CU_MEMORYTYPE_HOST; cp.srcHost = lvl; cp.srcPitch = w * 4;
      cp.dstMemoryType = CU_MEMORYTYPE_ARRAY; cp.dstArray = lv;
      cp.WidthInBytes = w * 4; cp.Height = w;
      CK(cuMemcpy2D(&cp));
    }
    CUDA_RESOURCE_DESC mr;
    memset(&mr, 0, sizeof mr);
    mr.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mr.res.mipmap.hMipmappedArray = mip;
    CUDA_TEXTURE_DESC mt;
    memset(&mt, 0, sizeof mt);
    mt.addressMode[0] = mt.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    mt.flags = CU_TRSF_NORMALIZED_COORDINATES;
    mt.maxMipmapLevelClamp = 6.0f;
    const float lods[8] = {-0.2966f, 0.49609375f, 0.5f, 1.25f, 2.75f, 5.5f, 9.0f, 0.05f};
    const float point[8] = {1, 1, 2, 2, 8, 64, 64, 1};
    const float linear[8] = {1, 1.49609375f, 1.5f, 2.5f, 7.0f, 48, 64, 1.046875f};
    const float biased[8] = {1.00390625f, 1.79296875f, 1.796875f, 3.09375f, 8.375f, 57.5f, 64, 1.34375f};
    CUdeviceptr dl, dv;
    CK(cuMemAlloc(&dl, sizeof lods));
    CK(cuMemAlloc(&dv, sizeof lods));
    CK(cuMemcpyHtoD(dl, lods, sizeof lods));
    CUmodule mm;
    CK(cuModuleLoadData(&mm, mipPtx));
    CUfunction mk;
    CK(cuModuleGetFunction(&mk, mm, "mipk"));
    const float *want[3] = {point, linear, biased};
    ok = 1;
    for (int c = 0; c < 3; c++) {
      mt.mipmapFilterMode = c ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;
      mt.mipmapLevelBias = c == 2 ? 0.3f : 0.0f;
      CUtexObject mtex;
      CK(cuTexObjectCreate(&mtex, &mr, &mt, NULL));
      void *margs[] = {&mtex, &dv, &dl};
      CK(cuLaunchKernel(mk, 1, 1, 1, 8, 1, 1, 0, NULL, margs, NULL));
      CK(cuCtxSynchronize());
      float got[8];
      CK(cuMemcpyDtoH(got, dv, sizeof got));
      for (int k = 0; k < 8; k++) ok = ok && got[k] == want[c][k];
      CK(cuTexObjectDestroy(mtex));
    }
    // Two levels of 0x3f0fab54 and 0x3faac74f blended at LOD 1/4: an exact tie, 0x3f412427.
    CUDA_ARRAY3D_DESCRIPTOR td2 = {64, 64, 0, CU_AD_FORMAT_FLOAT, 1, 0};
    CUmipmappedArray mip2;
    CK(cuMipmappedArrayCreate(&mip2, &td2, 2));
    const uint32_t two[2] = {0x3f0fab54u, 0x3faac74fu};
    for (unsigned k = 0; k < 2; k++) {
      CUarray lv;
      CK(cuMipmappedArrayGetLevel(&lv, mip2, k));
      unsigned w = 64 >> k;
      uint32_t lvl[64 * 64];
      for (unsigned e = 0; e < w * w; e++) lvl[e] = two[k];
      CUDA_MEMCPY2D cp;
      memset(&cp, 0, sizeof cp);
      cp.srcMemoryType = CU_MEMORYTYPE_HOST; cp.srcHost = lvl; cp.srcPitch = w * 4;
      cp.dstMemoryType = CU_MEMORYTYPE_ARRAY; cp.dstArray = lv;
      cp.WidthInBytes = w * 4; cp.Height = w;
      CK(cuMemcpy2D(&cp));
    }
    mr.res.mipmap.hMipmappedArray = mip2;
    mt.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
    mt.mipmapLevelBias = 0.0f;
    mt.maxMipmapLevelClamp = 1.0f;
    CUtexObject ttex;
    CK(cuTexObjectCreate(&ttex, &mr, &mt, NULL));
    const float quarter[8] = {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f};
    CK(cuMemcpyHtoD(dl, quarter, sizeof quarter));
    void *targs[] = {&ttex, &dv, &dl};
    CK(cuLaunchKernel(mk, 1, 1, 1, 8, 1, 1, 0, NULL, targs, NULL));
    CK(cuCtxSynchronize());
    uint32_t tie[8];
    CK(cuMemcpyDtoH(tie, dv, sizeof tie));
    ok = ok && tie[0] == 0x3f412427u;
    CK(cuTexObjectDestroy(ttex));
    CK(cuMipmappedArrayDestroy(mip2));
    // A view of levels 2 to 4 of the first array: its level 0 is the array's level 2, and LODs past
    // its last read that (a GH200's results at these LODs).
    CUDA_RESOURCE_VIEW_DESC mv;
    memset(&mv, 0, sizeof mv);
    mv.format = CU_RES_VIEW_FORMAT_FLOAT_1X32;
    mv.width = mv.height = 64;
    mv.firstMipmapLevel = 2;
    mv.lastMipmapLevel = 4;
    mr.res.mipmap.hMipmappedArray = mip;
    mt.maxMipmapLevelClamp = 6.0f;
    const float viewLods[8] = {-1, 0, 0.25f, 0.5f, 1, 1.5f, 2, 2.5f};
    const float viewWant[2][8] = {{4, 4, 4, 8, 8, 16, 16, 16}, {4, 4, 5, 6, 8, 12, 16, 16}};
    CK(cuMemcpyHtoD(dl, viewLods, sizeof viewLods));
    for (int c = 0; c < 2; c++) {
      mt.mipmapFilterMode = c ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;
      CUtexObject vtex;
      CK(cuTexObjectCreate(&vtex, &mr, &mt, &mv));
      void *vargs[] = {&vtex, &dv, &dl};
      CK(cuLaunchKernel(mk, 1, 1, 1, 8, 1, 1, 0, NULL, vargs, NULL));
      CK(cuCtxSynchronize());
      float got[8];
      CK(cuMemcpyDtoH(got, dv, sizeof got));
      for (int k = 0; k < 8; k++) ok = ok && got[k] == viewWant[c][k];
      CK(cuTexObjectDestroy(vtex));
    }
    mv.width = 16;
    CUtexObject badView;
    ok = ok && cuTexObjectCreate(&badView, &mr, &mt, &mv) == CUDA_ERROR_INVALID_VALUE;
    check("a mipmapped array's explicit-LOD fetch reads the nearest level or blends two, with bias and clamps, and through a view of some levels", ok);
    CK(cuModuleUnload(mm));
    CK(cuMemFree(dl));
    CK(cuMemFree(dv));
    CK(cuMipmappedArrayDestroy(mip));

    // Integer fetches by level and gathers from a mipmapped array (64 by 64 floats, 7 levels, texel
    // (level k, x, y) 1000 k + 64 y + x): an integer fetch reads the level it names, its
    // coordinates clamped, and a level past the last reads zero; a gather reads level 0 (a GH200's
    // and an A100's results).
    static const char *intPtx =
      ".version 8.0\n.target sm_80\n.address_size 64\n"
      ".visible .entry intk(.param .u64 tex_, .param .u64 in_, .param .u64 out_) {\n"
      "  .reg .b64 t, p, o, a; .reg .b32 i, x, y, k, u, v, r0, r1, r2, r3, g0, g1, g2, g3;\n"
      "  ld.param.u64 t, [tex_]; ld.param.u64 p, [in_]; ld.param.u64 o, [out_];\n"
      "  mov.u32 i, %tid.x; mul.wide.u32 a, i, 20; add.s64 p, p, a; mul.wide.u32 a, i, 20; add.s64 o, o, a;\n"
      "  ld.global.b32 x, [p]; ld.global.b32 y, [p+4]; ld.global.b32 k, [p+8];\n"
      "  ld.global.b32 u, [p+12]; ld.global.b32 v, [p+16];\n"
      "  tex.level.2d.v4.f32.s32 {r0, r1, r2, r3}, [t, {x, y}], k;\n"
      "  tld4.r.2d.v4.f32.f32 {g0, g1, g2, g3}, [t, {u, v}];\n"
      "  st.global.b32 [o], r0; st.global.b32 [o+4], g0; st.global.b32 [o+8], g1;\n"
      "  st.global.b32 [o+12], g2; st.global.b32 [o+16], g3;\n"
      "  ret;\n}\n";
    CUDA_ARRAY3D_DESCRIPTOR id3 = {64, 64, 0, CU_AD_FORMAT_FLOAT, 1, 0};
    CUmipmappedArray imip;
    CK(cuMipmappedArrayCreate(&imip, &id3, 7));
    for (unsigned k = 0; k < 7; k++) {
      CUarray lv;
      CK(cuMipmappedArrayGetLevel(&lv, imip, k));
      unsigned w = 64 >> k;
      static float ivals[64 * 64];
      for (unsigned y = 0; y < w; y++)
        for (unsigned x = 0; x < w; x++) ivals[y * w + x] = 1000.0f * k + 64.0f * y + x;
      CUDA_MEMCPY2D cp;
      memset(&cp, 0, sizeof cp);
      cp.srcMemoryType = CU_MEMORYTYPE_HOST; cp.srcHost = ivals; cp.srcPitch = w * 4;
      cp.dstMemoryType = CU_MEMORYTYPE_ARRAY; cp.dstArray = lv;
      cp.WidthInBytes = w * 4; cp.Height = w;
      CK(cuMemcpy2D(&cp));
    }
    // (x, y, level, u, v); the integer fetch's and the gather's results.
    static const int ixy[4][3] = {{5, 3, 0}, {40, 33, 1}, {70, -2, 3}, {1, 0, 9}};
    static const float iuv[4][2] = {{0.5f, 0.5f}, {0.1f, 0.2f}, {0.73f, 0.31f}, {0.02f, 0.97f}};
    static const float iwant[4][5] = {{197, 2079, 2080, 2016, 2015}, {3015, 837, 838, 774, 773},
                                      {3007, 1326, 1327, 1263, 1262}, {0, 3968, 3969, 3905, 3904}};
    uint32_t iin[4][5];
    for (int c = 0; c < 4; c++) {
      for (int e = 0; e < 3; e++) iin[c][e] = (uint32_t)ixy[c][e];
      memcpy(&iin[c][3], iuv[c], 8);
    }
    CUdeviceptr iinD, ioutD;
    CK(cuMemAlloc(&iinD, sizeof iin));
    CK(cuMemAlloc(&ioutD, sizeof iin));
    CK(cuMemcpyHtoD(iinD, iin, sizeof iin));
    mr.res.mipmap.hMipmappedArray = imip;
    mt.filterMode = CU_TR_FILTER_MODE_POINT;
    mt.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
    mt.mipmapLevelBias = 0.0f;
    mt.maxMipmapLevelClamp = 6.0f;
    CUtexObject itex;
    CK(cuTexObjectCreate(&itex, &mr, &mt, NULL));
    CUmodule im;
    CK(cuModuleLoadData(&im, intPtx));
    CUfunction ik;
    CK(cuModuleGetFunction(&ik, im, "intk"));
    void *iargs[] = {&itex, &iinD, &ioutD};
    CK(cuLaunchKernel(ik, 1, 1, 1, 4, 1, 1, 0, NULL, iargs, NULL));
    CK(cuCtxSynchronize());
    float iout[4][5];
    CK(cuMemcpyDtoH(iout, ioutD, sizeof iout));
    int iok = 1;
    for (int c = 0; c < 4; c++)
      for (int e = 0; e < 5; e++) {
        iok = iok && iout[c][e] == iwant[c][e];
        if (iout[c][e] != iwant[c][e]) printf("mip int/gather case %d.%d: %g, a GH200 %g\n", c, e, iout[c][e], iwant[c][e]);
      }
    check("a mipmapped array's integer fetch reads the level it names, and a gather level 0", iok);
    CK(cuTexObjectDestroy(itex));
    CK(cuModuleUnload(im));
    CK(cuMemFree(iinD));
    CK(cuMemFree(ioutD));
    CK(cuMipmappedArrayDestroy(imip));

    // Linear texels between linearly filtered levels: two levels of 64 by 64 and 32 by 32 values
    // from a generator, fetched at (u, v, LOD) where a GH200 returns `want` (the first two clamped
    // past the left edge of level 1).
    static const char *triPtx =
      ".version 8.0\n.target sm_80\n.address_size 64\n"
      ".visible .entry trik(.param .u64 tex_, .param .u64 out_, .param .u64 in_) {\n"
      "  .reg .b64 t, o, p, a; .reg .b32 i; .reg .f32 u, v, lod, r0, r1, r2, r3;\n"
      "  ld.param.u64 t, [tex_]; ld.param.u64 o, [out_]; ld.param.u64 p, [in_];\n"
      "  mov.u32 i, %tid.x; mul.wide.u32 a, i, 12; add.s64 p, p, a; mul.wide.u32 a, i, 4; add.s64 o, o, a;\n"
      "  ld.global.f32 u, [p]; ld.global.f32 v, [p+4]; ld.global.f32 lod, [p+8];\n"
      "  tex.level.2d.v4.f32.f32 {r0, r1, r2, r3}, [t, {u, v}], lod;\n"
      "  st.global.f32 [o], r0;\n"
      "  ret;\n}\n";
    static const uint32_t tri[6][4] = {
      {0x3c54fdf4, 0x3f7851ec, 0x3c800000, 0x4047afb4}, {0x3c54fdf4, 0x3f7851ec, 0x3d000000, 0x4045bddd},
      {0x3f3902de, 0x3ea6f695, 0x3e100000, 0x40064974}, {0x3d6ecbfc, 0x3f6dcc64, 0x3f180000, 0x400a9954},
      {0x3e18793e, 0x3f58c155, 0x3f600000, 0x400301e2}, {0x3e5652be, 0x3f4ab9f6, 0x3f540000, 0x3fc4b089}};
    CUmipmappedArray mip3;
    CK(cuMipmappedArrayCreate(&mip3, &td2, 2));
    uint32_t rng = 12345;
    for (unsigned k = 0; k < 2; k++) {
      CUarray lv;
      CK(cuMipmappedArrayGetLevel(&lv, mip3, k));
      unsigned w = 64 >> k;
      float lvl[64 * 64];
      for (unsigned e = 0; e <= w * w; e++) {
        rng = rng * 1664525u + 1013904223u;
        if (e) lvl[e - 1] = 0.5f + (rng >> 8) / 16777216.0f * 3.0f;
      }
      CUDA_MEMCPY2D cp;
      memset(&cp, 0, sizeof cp);
      cp.srcMemoryType = CU_MEMORYTYPE_HOST; cp.srcHost = lvl; cp.srcPitch = w * 4;
      cp.dstMemoryType = CU_MEMORYTYPE_ARRAY; cp.dstArray = lv;
      cp.WidthInBytes = w * 4; cp.Height = w;
      CK(cuMemcpy2D(&cp));
    }
    mr.res.mipmap.hMipmappedArray = mip3;
    mt.filterMode = CU_TR_FILTER_MODE_LINEAR;
    CK(cuTexObjectCreate(&ttex, &mr, &mt, NULL));
    uint32_t tin[18], tout[6];
    for (int c = 0; c < 6; c++) for (int e = 0; e < 3; e++) tin[3 * c + e] = tri[c][e];
    CUdeviceptr din, dout;
    CK(cuMemAlloc(&din, sizeof tin));
    CK(cuMemAlloc(&dout, sizeof tout));
    CK(cuMemcpyHtoD(din, tin, sizeof tin));
    CUmodule tm;
    CK(cuModuleLoadData(&tm, triPtx));
    CUfunction tk;
    CK(cuModuleGetFunction(&tk, tm, "trik"));
    void *triArgs[] = {&ttex, &dout, &din};
    CK(cuLaunchKernel(tk, 1, 1, 1, 6, 1, 1, 0, NULL, triArgs, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(tout, dout, sizeof tout));
    int tok = 1;
    for (int c = 0; c < 6; c++) {
      tok = tok && tout[c] == tri[c][3];
      if (tout[c] != tri[c][3]) printf("trilinear case %d: %08x, a GH200 %08x\n", c, tout[c], tri[c][3]);
    }
    check("linear filtering between linearly filtered mipmap levels weighs texels as a GH200 does", tok);
    CK(cuTexObjectDestroy(ttex));
    CK(cuModuleUnload(tm));
    CK(cuMemFree(din));
    CK(cuMemFree(dout));
    CK(cuMipmappedArrayDestroy(mip3));

    // Layered and cubemap mipmapped arrays: 32 by 32, 6 levels, 3 layers or 6 faces, texel (level
    // k, layer or face l, x, y) 1000 l + 100 k + 4 y + x; explicit-LOD fetches, linear between
    // levels, where a GH200 returns `want`: a layered fetch (layer, x, 1 - x) and a cubemap fetch
    // of the direction (2x - 1, 0.4 - x, 0.3) with the face's axis set to +-1, at each LOD.
    static const char *layPtx =
      ".version 8.0\n.target sm_80\n.address_size 64\n"
      ".visible .entry layk(.param .u64 tex_, .param .u64 in_, .param .u64 out_) {\n"
      "  .reg .b64 t, p, o, a; .reg .b32 i, l, x, y, lod, r0, r1, r2, r3;\n"
      "  ld.param.u64 t, [tex_]; ld.param.u64 p, [in_]; ld.param.u64 o, [out_];\n"
      "  mov.u32 i, %tid.x; mul.wide.u32 a, i, 16; add.s64 p, p, a; mul.wide.u32 a, i, 4; add.s64 o, o, a;\n"
      "  ld.global.b32 l, [p]; ld.global.b32 x, [p+4]; ld.global.b32 y, [p+8]; ld.global.b32 lod, [p+12];\n"
      "  tex.level.a2d.v4.f32.f32 {r0, r1, r2, r3}, [t, {l, x, y, y}], lod;\n"
      "  st.global.b32 [o], r0;\n"
      "  ret;\n}\n"
      ".visible .entry cubek(.param .u64 tex_, .param .u64 in_, .param .u64 out_) {\n"
      "  .reg .b64 t, p, o, a; .reg .b32 i, x, y, z, lod, r0, r1, r2, r3;\n"
      "  ld.param.u64 t, [tex_]; ld.param.u64 p, [in_]; ld.param.u64 o, [out_];\n"
      "  mov.u32 i, %tid.x; mul.wide.u32 a, i, 16; add.s64 p, p, a; mul.wide.u32 a, i, 4; add.s64 o, o, a;\n"
      "  ld.global.b32 x, [p]; ld.global.b32 y, [p+4]; ld.global.b32 z, [p+8]; ld.global.b32 lod, [p+12];\n"
      "  tex.level.cube.v4.f32.f32 {r0, r1, r2, r3}, [t, {x, y, z, z}], lod;\n"
      "  st.global.b32 [o], r0;\n"
      "  ret;\n}\n";
    static const float layArgs[2][4][3] = {
      {{0, 0.1f, 1.25f}, {2, 0.5f, 4}, {1, 0.83f, 2.75f}, {2, 0.1f, 0.5f}},
      {{4, 0.5f, 0}, {5, 0.5f, 1}, {0, 0.83f, 2.75f}, {1, 0.83f, 6}}};
    static const float layWant[2][4] = {{174.75f, 2405, 1279.75f, 2136}, {4084, 5140, 287.25f, 1500}};
    CUmodule lym;
    CK(cuModuleLoadData(&lym, layPtx));
    CUdeviceptr lyin, lyout;
    CK(cuMemAlloc(&lyin, 64));
    CK(cuMemAlloc(&lyout, 16));
    int lyok = 1;
    for (int c = 0; c < 2; c++) {
      CUDA_ARRAY3D_DESCRIPTOR ld = {32, 32, c ? 6 : 3, CU_AD_FORMAT_FLOAT, 1, c ? CUDA_ARRAY3D_CUBEMAP : CUDA_ARRAY3D_LAYERED};
      CUmipmappedArray lay;
      CK(cuMipmappedArrayCreate(&lay, &ld, 6));
      for (unsigned k = 0; k < 6; k++) {
        CUarray lv;
        CK(cuMipmappedArrayGetLevel(&lv, lay, k));
        unsigned w = 32 >> k;
        static float vals[6 * 32 * 32];
        for (unsigned l = 0; l < ld.Depth; l++)
          for (unsigned y = 0; y < w; y++)
            for (unsigned x = 0; x < w; x++) vals[(l * w + y) * w + x] = 1000.0f * l + 100.0f * k + 4.0f * y + x;
        CUDA_MEMCPY3D cp;
        memset(&cp, 0, sizeof cp);
        cp.srcMemoryType = CU_MEMORYTYPE_HOST; cp.srcHost = vals; cp.srcPitch = w * 4; cp.srcHeight = w;
        cp.dstMemoryType = CU_MEMORYTYPE_ARRAY; cp.dstArray = lv;
        cp.WidthInBytes = w * 4; cp.Height = w; cp.Depth = ld.Depth;
        CK(cuMemcpy3D(&cp));
      }
      float in[4][4];
      for (int e = 0; e < 4; e++) {
        float x = layArgs[c][e][1];
        if (c) {
          float d[3] = {2 * x - 1, 0.4f - x, 0.3f};
          d[(int)layArgs[c][e][0] / 2] = (int)layArgs[c][e][0] % 2 ? -1.0f : 1.0f;
          in[e][0] = d[0]; in[e][1] = d[1]; in[e][2] = d[2];
        } else {
          uint32_t l = (uint32_t)layArgs[c][e][0];
          memcpy(&in[e][0], &l, 4); in[e][1] = x; in[e][2] = 1 - x;
        }
        in[e][3] = layArgs[c][e][2];
      }
      CK(cuMemcpyHtoD(lyin, in, sizeof in));
      mr.res.mipmap.hMipmappedArray = lay;
      mt.filterMode = CU_TR_FILTER_MODE_POINT;
      mt.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
      mt.maxMipmapLevelClamp = 5.0f;
      CUtexObject lyt;
      CK(cuTexObjectCreate(&lyt, &mr, &mt, NULL));
      CUfunction lyk;
      CK(cuModuleGetFunction(&lyk, lym, c ? "cubek" : "layk"));
      void *lyargs[] = {&lyt, &lyin, &lyout};
      CK(cuLaunchKernel(lyk, 1, 1, 1, 4, 1, 1, 0, NULL, lyargs, NULL));
      CK(cuCtxSynchronize());
      float got[4];
      CK(cuMemcpyDtoH(got, lyout, sizeof got));
      for (int e = 0; e < 4; e++) {
        lyok = lyok && got[e] == layWant[c][e];
        if (got[e] != layWant[c][e]) printf("%s case %d: %g, a GH200 %g\n", c ? "cubemap" : "layered", e, got[e], layWant[c][e]);
      }
      CK(cuTexObjectDestroy(lyt));
      CK(cuMipmappedArrayDestroy(lay));
    }
    check("layered and cubemap mipmapped arrays read the level and layer or face a fetch names", lyok);
    CK(cuModuleUnload(lym));
    CK(cuMemFree(lyin));
    CK(cuMemFree(lyout));
  }
  CK(cuTexObjectDestroy(tex));
  CK(cuSurfObjectDestroy(surf));
  CK(cuArrayDestroy(tarr));
  CK(cuArrayDestroy(sarr));
  CK(cuMemFree(dout));
  CK(cuModuleUnload(m));
}

// Managed memory: the host writes an array through its pointer, a kernel (axpy, y = 2x + y)
// updates it in place from another managed array, the host reads the result through the same
// pointer after synchronizing, and a copy reads it too.
static void managedChecks(void) {
  enum { N = 3000 };
  CUdeviceptr x, y;
  CK(cuMemAllocManaged(&x, N * sizeof(float), CU_MEM_ATTACH_GLOBAL));
  CK(cuMemAllocManaged(&y, N * sizeof(float), CU_MEM_ATTACH_GLOBAL));
  float *hx = (float *)x, *hy = (float *)y;
  for (int i = 0; i < N; i++) { hx[i] = (float)i; hy[i] = 1.0f; }
  unsigned isManaged = 0;
  CK(cuPointerGetAttribute(&isManaged, CU_POINTER_ATTRIBUTE_IS_MANAGED, y));
  CUmodule m;
  CK(cuModuleLoadData(&m, ptx));
  CUfunction axpy;
  CK(cuModuleGetFunction(&axpy, m, "axpy"));
  float a = 2.0f;
  unsigned n = N;
  void *args[] = {&x, &y, &a, &n};
  CK(cuLaunchKernel(axpy, (N + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
  CK(cuCtxSynchronize());
  int ok = isManaged == 1;
  for (int i = 0; i < N; i++) ok = ok && hy[i] == 2.0f * (float)i + 1.0f;
  check("a kernel updates managed memory the host wrote, and the host reads it", ok);
  hx[5] = 100.0f;
  CK(cuLaunchKernel(axpy, (N + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
  CK(cuCtxSynchronize());
  float z[N];
  CK(cuMemcpyDtoH(z, y, sizeof z));
  ok = z[5] == 211.0f && z[6] == 25.0f;
  check("a host write between launches reaches the next, and a copy reads the result", ok);
  CK(cuMemFree(x));
  CK(cuMemFree(y));
  // Mapped page-locked memory: the kernel reads and writes it at its host address.
  float *px, *py;
  CK(cuMemHostAlloc((void **)&px, N * sizeof(float), CU_MEMHOSTALLOC_DEVICEMAP));
  CK(cuMemHostAlloc((void **)&py, N * sizeof(float), CU_MEMHOSTALLOC_DEVICEMAP));
  for (int i = 0; i < N; i++) { px[i] = (float)i; py[i] = 3.0f; }
  CUdeviceptr dx, dy;
  CK(cuMemHostGetDevicePointer(&dx, px, 0));
  CK(cuMemHostGetDevicePointer(&dy, py, 0));
  unsigned type = 0;
  CK(cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dy));
  void *margs[] = {&dx, &dy, &a, &n};
  CK(cuLaunchKernel(axpy, (N + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, margs, NULL));
  CK(cuCtxSynchronize());
  ok = type == CU_MEMORYTYPE_HOST && dx == (CUdeviceptr)px;
  for (int i = 0; i < N; i++) ok = ok && py[i] == 2.0f * (float)i + 3.0f;
  check("a kernel updates mapped page-locked memory, which the host reads", ok);
  CK(cuMemFreeHost(px));
  CK(cuMemFreeHost(py));
  CK(cuModuleUnload(m));
}

int main(int argc, char **argv) {
  // A deadlock in a broken driver must terminate the test process, including on the parent.
  alarm(120);
  setvbuf(stdout, NULL, _IOLBF, 0);
  CK(cuInit(0));
  {
    void *oldLookup, *newLookup, *missing = (void *)1;
    CUdriverProcAddressQueryResult status;
    CK(cuGetProcAddress("cuGetProcAddress", &oldLookup, 11030, 0, &status));
    CK(cuGetProcAddress("cuGetProcAddress", &newLookup, 12000, 0, &status));
    check("lookup selects distinct four- and five-argument ABIs", oldLookup != newLookup);
    CUresult (CUDAAPI *lookup)(const char *, void **, int, cuuint64_t) = oldLookup;
    void *init;
    CK(lookup("cuInit", &init, 2000, 0));
    CK(((CUresult (CUDAAPI *)(unsigned int))init)(0));
    CK(cuGetProcAddress("cuNvgpuAbsentFunction", &missing, CUDA_VERSION, 0, &status));
    check("missing symbol query succeeds with absent pointer and status",
          missing == NULL && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
    missing = (void *)1;
    check("CUDA 11 lookup reports missing symbols as an error",
          lookup("cuNvgpuAbsentFunction", &missing, 11030, 0) == CUDA_ERROR_NOT_FOUND && missing == NULL);
    CUstream stream;
    CK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));
    CUresult (CUDAAPI *capture)(CUstream, CUstreamCaptureStatus *, cuuint64_t *);
    CK(cuGetProcAddress("cuStreamGetCaptureInfo", (void **)&capture, 10010, 0, NULL));
    CUstreamCaptureStatus capturing;
    cuuint64_t id = 0;
    CK(capture(stream, &capturing, &id));
    check("CUDA 10 capture query ABI", capturing == CU_STREAM_CAPTURE_STATUS_ACTIVE && id != 0);
    CUgraph graph;
    CK(cuStreamEndCapture(stream, &graph));
    CK(cuGraphDestroy(graph));
    CUresult (CUDAAPI *getContext)(CUstream, CUcontext *, CUgreenCtx *);
    CK(cuGetProcAddress("cuStreamGetCtx", (void **)&getContext, 12050, 0, NULL));
    CUcontext context = NULL;
    CUgreenCtx green = (CUgreenCtx)1;
    CK(getContext(stream, &context, &green));
    check("CUDA 12.5 context query ABI", context != NULL && green == NULL);
    CK(cuStreamDestroy(stream));
  }
  if (argc == 3 && !strncmp(argv[1], "family-", 7)) {
    CUmodule family;
    CUresult r = cuModuleLoad(&family, argv[2]);
    if (!strcmp(argv[1], "family-ok")) {
      check("family PTX loads on its family", r == CUDA_SUCCESS);
      if (r == CUDA_SUCCESS) CK(cuModuleUnload(family));
    } else {
      check("family PTX refuses a different family", r == CUDA_ERROR_NO_BINARY_FOR_GPU);
    }
    return failures != 0;
  }
  if (argc > 1) {
    if (!strcmp(argv[1], "lookup")) return failures != 0;
    else if (!strcmp(argv[1], "failure")) failureDelivery();
    else if (!strcmp(argv[1], "graph")) graphSubmissionChecks();
    else if (!strcmp(argv[1], "legacy")) legacySubmissionChecks();
    else if (!strcmp(argv[1], "peer")) peerChecks();
    else if (!strcmp(argv[1], "managed")) {
      CUcontext c0;
      CUdevice d0;
      CK(cuDeviceGet(&d0, 0));
      CK(cuDevicePrimaryCtxRetain(&c0, d0));
      CK(cuCtxSetCurrent(c0));
      managedChecks();
    }
    else if (!strcmp(argv[1], "texture")) {
      CUcontext c0;
      CUdevice d0;
      CK(cuDeviceGet(&d0, 0));
      CK(cuDevicePrimaryCtxRetain(&c0, d0));
      CK(cuCtxSetCurrent(c0));
      textureChecks();
    }
    else return 2;
    return failures != 0;
  }

  CUdevice d;
  CK(cuDeviceGet(&d, 0));
  char name[128];
  int major, minor, sms;
  CK(cuDeviceGetName(name, sizeof name, d));
  CK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d));
  CK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d));
  CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d));
  size_t mem;
  CK(cuDeviceTotalMem(&mem, d));
  printf("%s: compute %d.%d, %d SMs, %zu bytes\n", name, major, minor, sms, mem);
  CUcontext c;
  CK(cuDevicePrimaryCtxRetain(&c, d));
  CK(cuCtxSetCurrent(c));
  CUmodule m;
  CK(cuModuleLoadData(&m, ptx));
  CUfunction axpy, bump;
  CK(cuModuleGetFunction(&axpy, m, "axpy"));
  CK(cuModuleGetFunction(&bump, m, "bump"));
  check("missing kernel refused", cuModuleGetFunction(&bump, m, "nonesuch") == CUDA_ERROR_NOT_FOUND);
  CK(cuModuleGetFunction(&bump, m, "bump"));

  // Module variables: the counter in device memory, the constant scale written by the host.
  CUdeviceptr counter, scale;
  size_t bytes;
  CK(cuModuleGetGlobal(&counter, &bytes, m, "counter"));
  check("cuModuleGetGlobal counter size", bytes == 4);
  CK(cuModuleGetGlobal(&scale, &bytes, m, "scale"));
  float initial;
  CK(cuMemcpyDtoH(&initial, scale, 4));
  check("__constant__ initializer in device memory", initial == 1.0f);
  float two = 2.0f;
  CK(cuMemcpyHtoD(scale, &two, 4));

  unsigned n = 1000;
  float *x = malloc(4 * n), *y = malloc(4 * n), *z = malloc(4 * n), a = 2.5f;
  for (unsigned i = 0; i < n; i++) { x[i] = (float)i * 0.5f; y[i] = 1.0f - (float)i; }
  CUdeviceptr dx, dy;
  CK(cuMemAlloc(&dx, 4 * n));
  CK(cuMemAlloc(&dy, 4 * n));
  CK(cuMemcpyHtoD(dx, x, 4 * n));
  CK(cuMemcpyHtoD(dy, y, 4 * n));
  void *args[] = {&dx, &dy, &a, &n};
  CK(cuLaunchKernel(axpy, (n + 127) / 128, 1, 1, 128, 1, 1, 0, 0, args, 0));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(z, dy, 4 * n));
  int bad = 0;
  for (unsigned i = 0; i < n; i++) if (z[i] != (a * 2.0f) * x[i] + y[i]) bad++;
  check("axpy with the host-written __constant__ scale", bad == 0);

  CK(cuMemsetD32(counter, 0, 1));
  CK(cuLaunchKernel(bump, 4, 1, 1, 64, 1, 1, 0, 0, NULL, 0));
  check("__device__ counter after 256 threads", readCounter(counter) == 256);

  // Two streams: s1 copies and computes, records e; s2 waits on e, sets, copies back, calls the
  // host; stream-ordered memory on s2.
  CUstream s1, s2;
  CUevent e, t0, t1;
  CK(cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING));
  CK(cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING));
  CK(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING));
  CK(cuEventCreate(&t0, 0));
  CK(cuEventCreate(&t1, 0));
  float *hp;
  CK(cuMemHostAlloc((void **)&hp, 4 * n, 0));
  for (unsigned i = 0; i < n; i++) hp[i] = (float)(i % 7);
  CUdeviceptr tmp;
  CK(cuMemAllocAsync(&tmp, 4 * n, s2));
  CK(cuEventRecord(t0, s1));
  CK(cuMemcpyHtoDAsync(dx, hp, 4 * n, s1));
  CK(cuMemsetD32Async(dy, 0x3f800000u, n, s1));  // 1.0f
  CK(cuLaunchKernel(axpy, (n + 127) / 128, 1, 1, 128, 1, 1, 0, s1, args, 0));
  CK(cuEventRecord(e, s1));
  CK(cuEventRecord(t1, s1));
  CK(cuStreamWaitEvent(s2, e, 0));
  CK(cuMemcpyDtoDAsync(tmp, dy, 4 * n, s2));
  CK(cuMemcpyDtoHAsync(hp, tmp, 4 * n, s2));
  CK(cuLaunchHostFunc(s2, onHost, NULL));
  CK(cuMemFreeAsync(tmp, s2));
  CK(cuStreamSynchronize(s2));
  CK(cuStreamSynchronize(s1));
  bad = 0;
  for (unsigned i = 0; i < n; i++) if (hp[i] != (a * 2.0f) * (float)(i % 7) + 1.0f) bad++;
  check("two streams ordered by an event", bad == 0 && hostCalls == 1);
  check("events complete", cuEventQuery(e) == CUDA_SUCCESS && cuStreamQuery(s1) == CUDA_SUCCESS);
  float ms = -1;
  CK(cuEventElapsedTime(&ms, t0, t1));
  check("elapsed time between recorded events", ms >= 0);

  // A graph built node by node: set the counter, two bumps, a copy of it to the host.
  CUgraph g;
  CK(cuGraphCreate(&g, 0));
  CUgraphNode set, k1, k2, cp;
  CUDA_MEMSET_NODE_PARAMS sp = {0};
  sp.dst = counter;
  sp.elementSize = 4;
  sp.width = 1;
  sp.height = 1;
  sp.value = 10;
  CK(cuGraphAddMemsetNode(&set, g, NULL, 0, &sp, c));
  CUDA_KERNEL_NODE_PARAMS kp = {0};
  kp.func = bump;
  kp.gridDimX = 2;
  kp.gridDimY = kp.gridDimZ = 1;
  kp.blockDimX = 32;
  kp.blockDimY = kp.blockDimZ = 1;
  CK(cuGraphAddKernelNode(&k1, g, &set, 1, &kp));
  CK(cuGraphAddKernelNode(&k2, g, &k1, 1, &kp));
  unsigned seen = 0;
  CUDA_MEMCPY3D cpy = {0};
  cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  cpy.srcDevice = counter;
  cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
  cpy.dstHost = &seen;
  cpy.WidthInBytes = 4;
  cpy.Height = 1;
  cpy.Depth = 1;
  CK(cuGraphAddMemcpyNode(&cp, g, &k2, 1, &cpy, c));
  CUgraphExec ge;
  CK(cuGraphInstantiateWithFlags(&ge, g, 0));
  CK(cuGraphLaunch(ge, s1));
  CK(cuStreamSynchronize(s1));
  check("built graph: set, two launches, copy out", seen == 10 + 128);
  seen = 0;
  CK(cuGraphLaunch(ge, s1));
  CK(cuStreamSynchronize(s1));
  check("built graph launched again", seen == 10 + 128);

  // A graph captured from s1, with a fork to s2 and a join through events.
  CUgraph cg;
  CUevent fork, join;
  CK(cuEventCreate(&fork, CU_EVENT_DISABLE_TIMING));
  CK(cuEventCreate(&join, CU_EVENT_DISABLE_TIMING));
  CK(cuMemsetD32(counter, 0, 1));
  CK(cuStreamBeginCapture(s1, CU_STREAM_CAPTURE_MODE_RELAXED));
  CUstreamCaptureStatus st;
  CK(cuStreamIsCapturing(s1, &st));
  check("stream capturing", st == CU_STREAM_CAPTURE_STATUS_ACTIVE);
  CK(cuLaunchKernel(bump, 1, 1, 1, 32, 1, 1, 0, s1, NULL, 0));
  CK(cuEventRecord(fork, s1));
  CK(cuStreamWaitEvent(s2, fork, 0));
  CK(cuLaunchKernel(bump, 1, 1, 1, 64, 1, 1, 0, s2, NULL, 0));
  CK(cuEventRecord(join, s2));
  CK(cuStreamWaitEvent(s1, join, 0));
  CK(cuStreamEndCapture(s1, &cg));
  check("capture ran nothing", readCounter(counter) == 0);
  size_t nodes = 0;
  CK(cuGraphGetNodes(cg, NULL, &nodes));
  check("captured graph has the two launches", nodes == 2);
  CUgraphExec cge;
  CK(cuGraphInstantiateWithFlags(&cge, cg, 0));
  for (int i = 0; i < 3; i++) CK(cuGraphLaunch(cge, s1));
  CK(cuStreamSynchronize(s1));
  check("captured graph launched three times", readCounter(counter) == 3 * 96);

  // A launch with attributes: a cluster shape where the part has clusters.
  if (major >= 9) {
    CUlaunchAttribute at[1];
    at[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    at[0].value.clusterDim.x = 2;
    at[0].value.clusterDim.y = 1;
    at[0].value.clusterDim.z = 1;
    CUlaunchConfig lc = {0};
    lc.gridDimX = 4;
    lc.gridDimY = lc.gridDimZ = 1;
    lc.blockDimX = 32;
    lc.blockDimY = lc.blockDimZ = 1;
    lc.hStream = s1;
    lc.attrs = at;
    lc.numAttrs = 1;
    CK(cuMemsetD32(counter, 0, 1));
    CK(cuLaunchKernelEx(&lc, bump, NULL, NULL));
    CK(cuStreamSynchronize(s1));
    check("cuLaunchKernelEx with 2-block clusters", readCounter(counter) == 128);
    lc.gridDimX = 3;
    check("grid not a whole number of clusters refused",
          cuLaunchKernelEx(&lc, bump, NULL, NULL) == CUDA_ERROR_INVALID_CLUSTER_SIZE);
  }

  // Tensor maps: a 2D float tile map encoded; its address replaced; a rank-0 map refused.
  CUtensorMap tm;
  cuuint64_t dims[2] = {256, 64}, strides[1] = {256 * 4};
  cuuint32_t box[2] = {32, 8}, es[2] = {1, 1};
  CK(cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void *)dx, dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CK(cuTensorMapReplaceAddress(&tm, (void *)dy));
  check("rank-0 tensor map refused",
        cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 0, (void *)dx, dims, strides, box, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
          == CUDA_ERROR_INVALID_VALUE);
  check("misaligned tensor map refused",
        cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void *)(dx + 4), dims, strides, box,
                               es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
          == CUDA_ERROR_INVALID_VALUE);

  // Occupancy.
  int blocks = 0, minGrid = 0, block = 0, regs = 0;
  CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, axpy, 128, 0));
  CK(cuOccupancyMaxPotentialBlockSize(&minGrid, &block, axpy, NULL, 0, 0));
  CK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, axpy));
  check("occupancy", blocks > 0 && block >= 32 && minGrid >= sms && regs > 0);

  // Pointer queries.
  unsigned type = 0;
  CK(cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dx));
  CUdeviceptr base = 0;
  CK(cuPointerGetAttribute(&base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, dx + 100));
  check("pointer attributes", type == CU_MEMORYTYPE_DEVICE && base == dx);

  // Ordering held up on purpose: stream `a` waits for a word another, nonblocking stream writes.
  // An event recorded on `a` and recorded again on `c` (which runs at once): a wait on `b`
  // enqueued between the two waits for `a`'s recording, not `c`'s; an executable launched on `a`
  // and destroyed at once still runs; a 32-bit NOR wait passes only when its 32 bits do.
  {
    CUstream a, b, c, w;
    CK(cuStreamCreate(&a, 0));
    CK(cuStreamCreate(&b, 0));
    CK(cuStreamCreate(&c, CU_STREAM_NON_BLOCKING));
    CK(cuStreamCreate(&w, CU_STREAM_NON_BLOCKING));
    CUdeviceptr flag;
    CK(cuMemAlloc(&flag, 4));
    CK(cuMemsetD32(flag, 0, 1));
    CK(cuMemsetD32(counter, 0, 1));
    CUevent e;
    CK(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING));
    CK(cuStreamWaitValue32(a, flag, 1, CU_STREAM_WAIT_VALUE_GEQ));
    CK(cuEventRecord(e, a));
    CK(cuStreamWaitEvent(b, e, 0));
    CK(cuLaunchHostFunc(b, onHost, NULL));
    CK(cuEventRecord(e, c));
    CK(cuStreamSynchronize(c));
    check("a wait holds for its own recording", cuStreamQuery(b) == CUDA_ERROR_NOT_READY);
    CUgraph lg;
    CUgraphExec lge;
    CUstream cap;
    CK(cuStreamCreate(&cap, 0));
    CK(cuStreamBeginCapture(cap, CU_STREAM_CAPTURE_MODE_RELAXED));
    CK(cuLaunchKernel(bump, 1, 1, 1, 32, 1, 1, 0, cap, NULL, 0));
    CK(cuStreamEndCapture(cap, &lg));
    CK(cuGraphInstantiateWithFlags(&lge, lg, 0));
    CK(cuGraphDestroy(lg));
    CK(cuGraphLaunch(lge, a));
    CK(cuGraphExecDestroy(lge));
    unsigned calls = hostCalls;
    CK(cuStreamWriteValue32(w, flag, 1, 0));
    CK(cuStreamSynchronize(b));
    CK(cuStreamSynchronize(a));
    check("the wait passes once its recording runs", hostCalls == calls + 1);
    check("a destroyed executable's queued launch runs", readCounter(counter) == 32);
    // NOR on 32 bits: ~(1 | 0xfffffffe) is 0 in 32 bits, so the wait holds until the word is 0.
    CK(cuStreamWriteValue32(w, flag, 1, 0));
    CK(cuStreamSynchronize(w));
    CK(cuStreamWaitValue32(c, flag, 0xfffffffeu, CU_STREAM_WAIT_VALUE_NOR));
    check("a 32-bit NOR wait holds", cuStreamQuery(c) == CUDA_ERROR_NOT_READY);
    CK(cuStreamWriteValue32(w, flag, 0, 0));
    CK(cuStreamSynchronize(c));
    CK(cuEventDestroy(e));
    CK(cuStreamDestroy(cap));
    // A destroyed stream with work still queued is waited for by the context.
    CK(cuStreamWaitValue32(a, flag, 2, CU_STREAM_WAIT_VALUE_GEQ));
    CK(cuLaunchHostFunc(a, onHost, NULL));
    CK(cuStreamDestroy(a));
    calls = hostCalls;
    CK(cuStreamWriteValue32(w, flag, 2, 0));
    CK(cuCtxSynchronize());
    check("a destroyed stream's work is synchronized", hostCalls == calls + 1);
    CK(cuStreamDestroy(b));
    CK(cuStreamDestroy(c));
    CK(cuStreamDestroy(w));
    CK(cuMemFree(flag));
  }

  // A batch memory-operation node: a write, a wait it satisfies and a 64-bit write, in order;
  // its parameters read back and replaced in the executable; a captured value write becomes one.
  {
    CUdeviceptr words;
    CK(cuMemAlloc(&words, 16));
    CK(cuMemsetD32(words, 0, 4));
    CUstreamBatchMemOpParams ops[3];
    memset(ops, 0, sizeof ops);
    ops[0].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    ops[0].writeValue.address = words;
    ops[0].writeValue.value = 7;
    ops[1].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    ops[1].waitValue.address = words;
    ops[1].waitValue.value = 7;
    ops[1].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    ops[2].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
    ops[2].writeValue.address = words + 8;
    ops[2].writeValue.value64 = 0x123456789ull;
    CUDA_BATCH_MEM_OP_NODE_PARAMS bp = {NULL, 3, ops, 0};
    CUgraph bg;
    CUgraphNode bn;
    CK(cuGraphCreate(&bg, 0));
    CK(cuGraphAddBatchMemOpNode(&bn, bg, NULL, 0, &bp));
    CUgraphNodeType type;
    CK(cuGraphNodeGetType(bn, &type));
    CUDA_BATCH_MEM_OP_NODE_PARAMS got;
    CK(cuGraphBatchMemOpNodeGetParams(bn, &got));
    CUgraphExec be;
    CK(cuGraphInstantiateWithFlags(&be, bg, 0));
    CK(cuGraphLaunch(be, NULL));
    CK(cuCtxSynchronize());
    uint32_t w[4];
    CK(cuMemcpyDtoH(w, words, sizeof w));
    check("a batch node writes, waits and writes in order",
          type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP && got.count == 3 && w[0] == 7 && w[2] == 0x23456789u && w[3] == 1);
    ops[0].writeValue.value = 5;
    ops[1].waitValue.value = 5;
    CK(cuGraphExecBatchMemOpNodeSetParams(be, bn, &bp));
    CK(cuGraphLaunch(be, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(w, words, sizeof w));
    check("an executable's batch node takes new operations", w[0] == 5);
    CK(cuGraphExecDestroy(be));
    CK(cuGraphDestroy(bg));
    CUstream cs;
    CK(cuStreamCreate(&cs, CU_STREAM_NON_BLOCKING));
    CK(cuStreamBeginCapture(cs, CU_STREAM_CAPTURE_MODE_RELAXED));
    CK(cuStreamWriteValue32(cs, words, 3, 0));
    CK(cuStreamEndCapture(cs, &bg));
    size_t nn = 1;
    CK(cuGraphGetNodes(bg, &bn, &nn));
    CK(cuGraphNodeGetType(bn, &type));
    check("a captured value write is a batch node", nn == 1 && type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP);
    CK(cuGraphDestroy(bg));
    CK(cuStreamDestroy(cs));
    CK(cuMemFree(words));
  }

  // Memory allocated during capture belongs to the graph. Free nodes release its contents,
  // while graph/executable/scheduled-launch ownership keeps its address until replay ends.
  {
    CUgraph ag;
    CUgraphExec ae;
    CUdeviceptr ap, other;
    unsigned got = 0;
    CK(cuStreamBeginCapture(s1, CU_STREAM_CAPTURE_MODE_RELAXED));
    CK(cuMemAllocAsync(&ap, 4096, s1));
    CK(cuMemsetD32Async(ap, 123, 1, s1));
    CK(cuMemcpyDtoHAsync(&got, ap, 4, s1));
    CK(cuMemFreeAsync(ap, s1));
    CK(cuStreamEndCapture(s1, &ag));
    check("captured allocation does not execute its copy", got == 0);
    CK(cuGraphInstantiateWithFlags(&ae, ag, 0));
    CK(cuGraphDestroy(ag));
    CK(cuGraphLaunch(ae, s1));
    CK(cuStreamSynchronize(s1));
    check("captured allocation and free execute", got == 123);
    CK(cuMemAlloc(&other, 4096));
    check("graph replay reserves the allocation address", other != ap);
    got = 0;
    CK(cuGraphLaunch(ae, s1));
    CK(cuStreamSynchronize(s1));
    check("captured allocation replays after its free", got == 123);
    CK(cuGraphExecDestroy(ae));
    CK(cuMemFree(other));
    CK(cuMemAlloc(&other, 4096));
    check("last graph owner releases its allocation address", other == ap);
    CK(cuMemFree(other));
  }

  // Ordinary context streams have no green context.
  {
    CUgreenCtx green = (CUgreenCtx)1;
    CK(cuStreamGetGreenCtx(s1, &green));
    check("ordinary stream has no green context", green == NULL);
  }

  CK(cuGraphExecDestroy(ge));
  CK(cuGraphExecDestroy(cge));
  CK(cuGraphDestroy(g));
  CK(cuGraphDestroy(cg));
  CK(cuStreamDestroy(s1));
  CK(cuStreamDestroy(s2));
  CK(cuMemFreeHost(hp));
  CK(cuMemFree(dx));
  CK(cuMemFree(dy));

  printf("driver API: %s (%d failed)\n", failures ? "FAIL" : "ok", failures);
  return failures != 0;
}
