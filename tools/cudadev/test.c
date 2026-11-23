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
  CK(cuTexObjectDestroy(tex));
  CK(cuSurfObjectDestroy(surf));
  CK(cuArrayDestroy(tarr));
  CK(cuArrayDestroy(sarr));
  CK(cuMemFree(dout));
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
