// The driver's older ABIs that the current one replaced with a different signature or structure,
// still exported by the driver under their original names (a library resolving them by dlsym
// needs each to exist; an older runtime asks for them through cuGetProcAddress).
// Each converts its arguments and calls the current function. Older ABIs with the current one's
// signature are aliases of it (gen.py); the 32-bit ones of CUDA 3 and the rest are stubs naming
// themselves. A translation unit without cuda.h, whose macros rename these names to the current
// ones: the types are declared here as cuda.h's internal section declares the old functions.

#include <stddef.h>
#include <string.h>

typedef int CUresult;
typedef void *CUgraph, *CUgraphNode, *CUgraphExec, *CUfunction, *CUkernel, *CUcontext;

// CUDA_KERNEL_NODE_PARAMS_v1 (CUDA 10-11) and _v2 (CUDA 12: the kernel and context added).
typedef struct {
  CUfunction func;
  unsigned gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
  void **kernelParams, **extra;
} KernelNodeParams1;
typedef struct {
  CUfunction func;
  unsigned gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
  void **kernelParams, **extra;
  CUkernel kern;
  CUcontext ctx;
} KernelNodeParams2;

// The current functions, by their exported names.
extern CUresult cuGraphInstantiateWithFlags(CUgraphExec *e, CUgraph g, unsigned long long flags);
extern CUresult cuGraphAddKernelNode_v2(CUgraphNode *n, CUgraph g, const CUgraphNode *deps, size_t nd,
                                        const KernelNodeParams2 *p);
extern CUresult cuGraphKernelNodeGetParams_v2(CUgraphNode n, KernelNodeParams2 *p);
extern CUresult cuGraphKernelNodeSetParams_v2(CUgraphNode n, const KernelNodeParams2 *p);
extern CUresult cuGraphExecKernelNodeSetParams_v2(CUgraphExec e, CUgraphNode n, const KernelNodeParams2 *p);

static KernelNodeParams2 widen(const KernelNodeParams1 *p) {
  KernelNodeParams2 q;
  memset(&q, 0, sizeof q);
  memcpy(&q, p, sizeof *p);
  return q;
}

// CUDA 10's instantiation: an error node and a log the device never fills (instantiation of a
// valid graph does not fail here; an invalid one fails with the current form's error).
CUresult cuGraphInstantiate(CUgraphExec *e, CUgraph g, CUgraphNode *err, char *log, size_t n) {
  if (err) *err = NULL;
  if (log && n) log[0] = 0;
  return cuGraphInstantiateWithFlags(e, g, 0);
}
CUresult cuGraphInstantiate_v2(CUgraphExec *e, CUgraph g, CUgraphNode *err, char *log, size_t n) {
  return cuGraphInstantiate(e, g, err, log, n);
}
CUresult cuGraphAddKernelNode(CUgraphNode *n, CUgraph g, const CUgraphNode *deps, size_t nd,
                              const KernelNodeParams1 *p) {
  KernelNodeParams2 q = widen(p);
  return cuGraphAddKernelNode_v2(n, g, deps, nd, &q);
}
CUresult cuGraphKernelNodeGetParams(CUgraphNode n, KernelNodeParams1 *p) {
  KernelNodeParams2 q;
  CUresult r = cuGraphKernelNodeGetParams_v2(n, &q);
  if (r == 0) memcpy(p, &q, sizeof *p);
  return r;
}
CUresult cuGraphKernelNodeSetParams(CUgraphNode n, const KernelNodeParams1 *p) {
  KernelNodeParams2 q = widen(p);
  return cuGraphKernelNodeSetParams_v2(n, &q);
}
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec e, CUgraphNode n, const KernelNodeParams1 *p) {
  KernelNodeParams2 q = widen(p);
  return cuGraphExecKernelNodeSetParams_v2(e, n, &q);
}

// CUDA 11's separate outputs; CUDA 12 combines them into CUgraphExecUpdateResultInfo.
typedef struct {
  int result;
  CUgraphNode errorNode, errorFromNode;
} GraphUpdateInfo;
extern CUresult cuGraphExecUpdate_v2(CUgraphExec e, CUgraph g, GraphUpdateInfo *info);
CUresult cuGraphExecUpdate(CUgraphExec e, CUgraph g, CUgraphNode *errorNode, int *result) {
  GraphUpdateInfo info = {0};
  CUresult r = cuGraphExecUpdate_v2(e, g, &info);
  if (errorNode) *errorNode = info.errorNode;
  if (result) *result = info.result;
  return r;
}
