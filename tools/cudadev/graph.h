// Graphs and stream capture of the emulated driver. Included by cuda.c, after stream.h.
//
// A graph is its nodes, each a piece of work (`Op`, stream.h) or a child graph, with the nodes
// it depends on. Instantiation orders the nodes so every node follows its dependencies (Kahn's
// order, ties in creation order); launching the executable graph enqueues it on the stream as one
// piece of work that runs its nodes in that order, which is an execution CUDA allows. Capture
// turns work submitted to a capturing stream into nodes: a stream's next node depends on its
// last, an event recorded in a capture carries the nodes before it, and a stream that waits on
// such an event joins the capture.
//
// Conditional nodes run a body graph if, while, or by which value a conditional handle holds when
// the node runs. A handle is a word of device memory the driver allocates; its value is the
// word's address, which the device runtime's `cudaGraphSetConditional(handle, value)` stores to
// (NV.SASS.Sem); a handle made with CU_GRAPH_COND_ASSIGN_DEFAULT is reset to its default each
// time the graph runs.

// Graph allocations reserve their virtual address across replay. Graphs, executables and
// queued snapshots share that reservation; a free node releases its contents, and the final
// owner also releases the address. This is separate from ordinary stream allocations.
typedef struct GraphMemory {
  CUdeviceptr ptr;
  int refs;
} GraphMemory;
static GraphMemory *graphMemory(CUdeviceptr ptr) {
  GraphMemory *m = malloc(sizeof *m);
  *m = (GraphMemory){ptr, 1};
  pthread_mutex_lock(&lock);
  findAlloc(ptr)->graphOwned = 1;
  pthread_mutex_unlock(&lock);
  return m;
}
static void graphMemoryRetain(GraphMemory *m) {
  if (m) __atomic_add_fetch(&m->refs, 1, __ATOMIC_RELAXED);
}
static void graphMemoryRelease(GraphMemory *m) {
  if (!m || __atomic_sub_fetch(&m->refs, 1, __ATOMIC_ACQ_REL)) return;
  pthread_mutex_lock(&lock);
  findAlloc(m->ptr)->graphOwned = 0;
  freeRange(m->ptr, 0);
  pthread_mutex_unlock(&lock);
  free(m);
}

struct CUgraphNode_st {
  CUgraphNodeType type;
  Op *op;          // the work (kernel, copy, set, host, event, alloc, free, empty)
  CUgraph child;   // a child graph
  CUgraph owner;
  struct CUgraphNode_st *origin;  // the node this one was copied from (instantiation copies)
  // The node the copies began from: a child graph's node is copied when the child is added (its
  // owner's copy) and again when the owner is instantiated, and the caller holds the first.
  struct CUgraphNode_st *root;
  struct CUgraphNode_st **deps;
  size_t ndeps, depsCap;
  int enabled;
  size_t order;    // creation order
  // A kernel node's parameters as given (queries return them).
  CUDA_KERNEL_NODE_PARAMS_v2 kparams;
  unsigned cluster[3];
  CUdeviceptr allocPtr;  // an allocation node's memory
  size_t allocBytes;
  GraphMemory *memory;
  // A conditional node: its kind, handle and body graphs (which it owns).
  CUgraphConditionalNodeType condType;
  CUdeviceptr cond;
  CUgraph *bodies;
  unsigned nbodies;
};

typedef struct Cond {
  CUdeviceptr word;
  GraphMemory *memory;
  unsigned dflt;
  int assign;  // reset to `dflt` at each launch of the graph
} Cond;

struct CUgraph_st {
  CUgraphNode *nodes;
  size_t n, cap;
  Cond *conds;  // the conditional handles made for this graph
  size_t nconds;
};

static void pushNode(CUgraph g, CUgraphNode n) {
  if (g->n == g->cap) g->nodes = realloc(g->nodes, sizeof *g->nodes * (g->cap = g->cap ? 2 * g->cap : 8));
  n->order = g->n;
  g->nodes[g->n++] = n;
}

static void addDep(CUgraphNode n, CUgraphNode d) {
  for (size_t i = 0; i < n->ndeps; i++)
    if (n->deps[i] == d) return;
  if (n->ndeps == n->depsCap) n->deps = realloc(n->deps, sizeof *n->deps * (n->depsCap = n->depsCap ? 2 * n->depsCap : 4));
  n->deps[n->ndeps++] = d;
}

static CUgraphNode newNode(CUgraph g, CUgraphNodeType t, Op *op, const CUgraphNode *deps, size_t nd) {
  CUgraphNode n = calloc(1, sizeof *n);
  n->type = t;
  n->op = op;
  n->owner = g;
  n->enabled = 1;
  for (size_t i = 0; i < nd; i++) addDep(n, deps[i]);
  pushNode(g, n);
  return n;
}

CUresult CUDAAPI cuGraphCreate(CUgraph *g, unsigned int flags) {
  (void)flags;
  *g = calloc(1, sizeof(struct CUgraph_st));
  return CUDA_SUCCESS;
}

// Copy or clear the conditional handles, retaining the source before dropping old ownership.
// Updates use this recursively, so child graphs reset the handles their new nodes reference.
static void graphConditions(CUgraph dst, CUgraph src) {
  size_t n = src ? src->nconds : 0;
  Cond *conds = n ? malloc(sizeof *conds * n) : NULL;
  if (n) memcpy(conds, src->conds, sizeof *conds * n);
  for (size_t i = 0; i < n; i++) graphMemoryRetain(conds[i].memory);
  for (size_t i = 0; i < dst->nconds; i++) graphMemoryRelease(dst->conds[i].memory);
  free(dst->conds);
  dst->conds = conds;
  dst->nconds = n;
}

static void graphFree(CUgraph g) {
  for (size_t i = 0; i < g->n; i++) {
    CUgraphNode n = g->nodes[i];
    if (n->op) opFree(n->op);
    graphMemoryRelease(n->memory);
    if (n->child) graphFree(n->child);
    for (unsigned b = 0; b < n->nbodies; b++) graphFree(n->bodies[b]);
    free(n->bodies);
    free(n->deps);
    free(n);
  }
  graphConditions(g, NULL);
  free(g->nodes);
  free(g);
}
CUresult CUDAAPI cuGraphDestroy(CUgraph g) {
  graphFree(g);
  return CUDA_SUCCESS;
}

// A deep copy of `g` (child graphs too), each node knowing its original.
static CUgraph graphClone(CUgraph g) {
  CUgraph c = calloc(1, sizeof *c);
  for (size_t i = 0; i < g->n; i++) {
    CUgraphNode n = g->nodes[i], m = calloc(1, sizeof *m);
    *m = *n;
    m->owner = c;
    m->origin = n;
    m->root = n->root ? n->root : n;
    graphMemoryRetain(m->memory);
    m->op = n->op ? opCopy(n->op) : NULL;
    m->child = n->child ? graphClone(n->child) : NULL;
    if (n->nbodies) {
      m->bodies = malloc(sizeof *m->bodies * n->nbodies);
      for (unsigned b = 0; b < n->nbodies; b++) m->bodies[b] = graphClone(n->bodies[b]);
    }
    m->depsCap = n->ndeps;
    m->deps = malloc(sizeof *m->deps * (n->ndeps ? n->ndeps : 1));
    pushNode(c, m);
  }
  for (size_t i = 0; i < g->n; i++)
    for (size_t j = 0; j < g->nodes[i]->ndeps; j++) c->nodes[i]->deps[j] = c->nodes[g->nodes[i]->deps[j]->order];
  graphConditions(c, g);
  return c;
}
CUresult CUDAAPI cuGraphClone(CUgraph *out, CUgraph g) {
  *out = graphClone(g);
  return CUDA_SUCCESS;
}

// ---- Adding nodes ----

static CUresult kernelNodeOp(Op **o, const CUDA_KERNEL_NODE_PARAMS_v2 *p, const unsigned cluster[3]) {
  CUfunction f = p->func ? p->func : (CUfunction)p->kern;
  unsigned g[3] = {p->gridDimX, p->gridDimY, p->gridDimZ}, b[3] = {p->blockDimX, p->blockDimY, p->blockDimZ};
  return kernelOp(o, f, g, b, cluster, p->sharedMemBytes, p->kernelParams, p->extra);
}

static CUresult addKernel(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd,
                          const CUDA_KERNEL_NODE_PARAMS_v2 *p) {
  NEED_INIT();
  unsigned cl[3] = {0, 0, 0};
  Op *o;
  CUresult r = kernelNodeOp(&o, p, cl);
  if (r != CUDA_SUCCESS) return r;
  CUgraphNode n = newNode(g, CU_GRAPH_NODE_TYPE_KERNEL, o, deps, nd);
  n->kparams = *p;
  n->kparams.kernelParams = NULL;  // the arguments are copied into the node's parameter block
  n->kparams.extra = NULL;
  *out = n;
  return CUDA_SUCCESS;
}

#if CUDA_VERSION >= 12000
CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd,
                                      const CUDA_KERNEL_NODE_PARAMS_v2 *p) {
  return addKernel(out, g, deps, nd, p);
}
#endif

CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd,
                                      const CUDA_MEMCPY3D *c, CUcontext ctx) {
  (void)ctx;
  Op *o = copy3DOp(c);
  if (!o) return CUDA_ERROR_NOT_SUPPORTED;
  *out = newNode(g, CU_GRAPH_NODE_TYPE_MEMCPY, o, deps, nd);
  return CUDA_SUCCESS;
}

static Op *memsetNodeOp(CUdeviceptr dst, size_t pitch, unsigned v, unsigned elem, size_t w, size_t h) {
  Op *o = newOp(OP_SET);
  o->dst = dst;
  o->value = v;
  o->elem = elem;
  o->width = w;
  o->height = h ? h : 1;
  o->dpitch = pitch ? pitch : w * elem;
  return o;
}

CUresult CUDAAPI cuGraphAddMemsetNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd,
                                      const CUDA_MEMSET_NODE_PARAMS *p, CUcontext ctx) {
  (void)ctx;
  *out = newNode(g, CU_GRAPH_NODE_TYPE_MEMSET,
                 memsetNodeOp(p->dst, p->pitch, p->value, p->elementSize, p->width, p->height), deps, nd);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddHostNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd,
                                    const CUDA_HOST_NODE_PARAMS *p) {
  Op *o = newOp(OP_HOST);
  o->fn = p->fn;
  o->user = p->userData;
  *out = newNode(g, CU_GRAPH_NODE_TYPE_HOST, o, deps, nd);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphHostNodeGetParams(CUgraphNode n, CUDA_HOST_NODE_PARAMS *p) {
  if (n->type != CU_GRAPH_NODE_TYPE_HOST) return CUDA_ERROR_INVALID_VALUE;
  p->fn = n->op->fn;
  p->userData = n->op->user;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphHostNodeSetParams(CUgraphNode n, const CUDA_HOST_NODE_PARAMS *p) {
  if (n->type != CU_GRAPH_NODE_TYPE_HOST) return CUDA_ERROR_INVALID_VALUE;
  n->op->fn = p->fn;
  n->op->user = p->userData;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddEmptyNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd) {
  *out = newNode(g, CU_GRAPH_NODE_TYPE_EMPTY, newOp(OP_NOP), deps, nd);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphAddChildGraphNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps,
                                          size_t nd, CUgraph child) {
  CUgraphNode n = newNode(g, CU_GRAPH_NODE_TYPE_GRAPH, NULL, deps, nd);
  n->child = graphClone(child);
  *out = n;
  return CUDA_SUCCESS;
}

// An event node: a recording numbered when the graph runs (seq 0: the event's next), or a wait for
// the recording before the graph ran.
static Op *eventOp(OpKind k, CUevent e) {
  Op *o = newOp(k);
  o->ev = e;
  eventRetain(e);
  return o;
}

CUresult CUDAAPI cuGraphAddEventRecordNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps,
                                           size_t nd, CUevent e) {
  *out = newNode(g, CU_GRAPH_NODE_TYPE_EVENT_RECORD, eventOp(OP_RECORD, e), deps, nd);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphAddEventWaitNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps,
                                         size_t nd, CUevent e) {
  *out = newNode(g, CU_GRAPH_NODE_TYPE_WAIT_EVENT, eventOp(OP_WAIT, e), deps, nd);
  return CUDA_SUCCESS;
}

// An allocation node: its memory is the graph's from creation (its address is fixed then, as the
// driver's is), so running the node does nothing; the free node gives it back.
CUresult CUDAAPI cuGraphAddMemAllocNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps,
                                        size_t nd, CUDA_MEM_ALLOC_NODE_PARAMS *p) {
  NEED_INIT();
  CUdeviceptr a;
  CUresult r = allocRange(&a, p->bytesize, ALLOC_ALIGN, 0);
  if (r != CUDA_SUCCESS) return r;
  p->dptr = a;
  CUgraphNode n = newNode(g, CU_GRAPH_NODE_TYPE_MEM_ALLOC, newOp(OP_NOP), deps, nd);
  n->allocPtr = a;
  n->allocBytes = p->bytesize;
  n->memory = graphMemory(a);
  *out = n;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphAddMemFreeNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd,
                                       CUdeviceptr p) {
  Op *op = newOp(OP_FREE);
  op->dst = p;
  *out = newNode(g, CU_GRAPH_NODE_TYPE_MEM_FREE, op, deps, nd);
  return CUDA_SUCCESS;
}

// A batch memory-operation node: stream memory operations run in order (`memOpsOp`), as a
// capture of cuStreamWriteValue32 and its kin makes too.
CUresult CUDAAPI cuGraphAddBatchMemOpNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps,
                                          size_t nd, const CUDA_BATCH_MEM_OP_NODE_PARAMS *p) {
  if (!p || (p->count && !p->paramArray)) return CUDA_ERROR_INVALID_VALUE;
  *out = newNode(g, CU_GRAPH_NODE_TYPE_BATCH_MEM_OP, memOpsOp(p->paramArray, p->count), deps, nd);
  return CUDA_SUCCESS;
}
// Its operations, which the node keeps.
CUresult CUDAAPI cuGraphBatchMemOpNodeGetParams(CUgraphNode n, CUDA_BATCH_MEM_OP_NODE_PARAMS *p) {
  if (n->type != CU_GRAPH_NODE_TYPE_BATCH_MEM_OP) return CUDA_ERROR_INVALID_VALUE;
  cuCtxGetCurrent(&p->ctx);
  p->count = n->op->nmemops;
  p->paramArray = n->op->memops;
  p->flags = 0;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphBatchMemOpNodeSetParams(CUgraphNode n, const CUDA_BATCH_MEM_OP_NODE_PARAMS *p) {
  if (n->type != CU_GRAPH_NODE_TYPE_BATCH_MEM_OP || !p || (p->count && !p->paramArray)) return CUDA_ERROR_INVALID_VALUE;
  Op *o = memOpsOp(p->paramArray, p->count);
  opFree(n->op);
  n->op = o;
  return CUDA_SUCCESS;
}

#if CUDA_VERSION >= 12030
CUresult CUDAAPI cuGraphConditionalHandleCreate(CUgraphConditionalHandle *h, CUgraph g, CUcontext c,
                                                unsigned int dflt, unsigned int flags) {
  NEED_INIT();
  (void)c;
  CUdeviceptr word;
  CUresult r = allocRange(&word, 4, ALLOC_ALIGN, 0);
  if (r != CUDA_SUCCESS) return r;
  // Without ASSIGN_DEFAULT the value is undefined until a kernel sets it: zero here.
  unsigned v = flags & CU_GRAPH_COND_ASSIGN_DEFAULT ? dflt : 0;
  r = devWrite(word, &v, 4);
  if (r != CUDA_SUCCESS) {
    freeRange(word, 0);
    return r;
  }
  g->conds = realloc(g->conds, sizeof *g->conds * (g->nconds + 1));
  g->conds[g->nconds++] = (Cond){word, graphMemory(word), dflt, (flags & CU_GRAPH_COND_ASSIGN_DEFAULT) != 0};
  *h = (CUgraphConditionalHandle)word;
  return CUDA_SUCCESS;
}

// A conditional node: its body graphs made empty, for the program to fill; `phGraph_out` is set
// to the node's own array of them (CUDA's contract: valid while the node lives).
static CUresult addConditional(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd,
                               CUDA_CONDITIONAL_NODE_PARAMS *p) {
  unsigned size = p->size ? p->size : 1;
  if (p->type == CU_GRAPH_COND_TYPE_WHILE && size != 1) return CUDA_ERROR_INVALID_VALUE;
  CUgraphNode n = newNode(g, CU_GRAPH_NODE_TYPE_CONDITIONAL, NULL, deps, nd);
  n->condType = p->type;
  n->cond = (CUdeviceptr)p->handle;
  n->nbodies = size;
  n->bodies = malloc(sizeof *n->bodies * size);
  for (unsigned b = 0; b < size; b++) cuGraphCreate(&n->bodies[b], 0);
  p->phGraph_out = n->bodies;
  *out = n;
  return CUDA_SUCCESS;
}
#endif

// The generic form: a node of any type from its parameters (CUDA 12.2's form, without edge data).
static CUresult legacyGraphAddNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd,
                                   CUgraphNodeParams *p) {
  switch (p->type) {
    case CU_GRAPH_NODE_TYPE_KERNEL: {
      CUDA_KERNEL_NODE_PARAMS_v2 k;
      memcpy(&k, &p->kernel, sizeof k);
      return addKernel(out, g, deps, nd, &k);
    }
    case CU_GRAPH_NODE_TYPE_MEMCPY: return cuGraphAddMemcpyNode(out, g, deps, nd, &p->memcpy.copyParams, NULL);
    case CU_GRAPH_NODE_TYPE_MEMSET:
      *out = newNode(g, CU_GRAPH_NODE_TYPE_MEMSET,
                     memsetNodeOp(p->memset.dst, p->memset.pitch, p->memset.value, p->memset.elementSize,
                                  p->memset.width, p->memset.height), deps, nd);
      return CUDA_SUCCESS;
    case CU_GRAPH_NODE_TYPE_HOST: {
      CUDA_HOST_NODE_PARAMS h = {p->host.fn, p->host.userData};
      return cuGraphAddHostNode(out, g, deps, nd, &h);
    }
    case CU_GRAPH_NODE_TYPE_GRAPH: return cuGraphAddChildGraphNode(out, g, deps, nd, p->graph.graph);
    case CU_GRAPH_NODE_TYPE_EMPTY: return cuGraphAddEmptyNode(out, g, deps, nd);
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT: return cuGraphAddEventWaitNode(out, g, deps, nd, p->eventWait.event);
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD: return cuGraphAddEventRecordNode(out, g, deps, nd, p->eventRecord.event);
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC: {
      CUDA_MEM_ALLOC_NODE_PARAMS a;
      memset(&a, 0, sizeof a);
      a.bytesize = p->alloc.bytesize;
      CUresult r = cuGraphAddMemAllocNode(out, g, deps, nd, &a);
      p->alloc.dptr = a.dptr;
      return r;
    }
    case CU_GRAPH_NODE_TYPE_MEM_FREE: return cuGraphAddMemFreeNode(out, g, deps, nd, p->free.dptr);
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: {
      CUDA_BATCH_MEM_OP_NODE_PARAMS b = {.ctx = p->memOp.ctx, .count = p->memOp.count,
                                         .paramArray = p->memOp.paramArray, .flags = p->memOp.flags};
      return cuGraphAddBatchMemOpNode(out, g, deps, nd, &b);
    }
#if CUDA_VERSION >= 12030
    case CU_GRAPH_NODE_TYPE_CONDITIONAL: return addConditional(out, g, deps, nd, &p->conditional);
#endif
    default:
      note("graph node type %d is not modelled", (int)p->type);
      return CUDA_ERROR_NOT_SUPPORTED;
  }
}
CUresult CUDAAPI cuGraphAddNode_v2(CUgraphNode *out, CUgraph g, const CUgraphNode *deps,
                                   const CUgraphEdgeData *edges, size_t nd, CUgraphNodeParams *p) {
  (void)edges;  // edge kinds order programmatic launches, which run in order here
  return legacyGraphAddNode(out, g, deps, nd, p);
}
#if CUDA_VERSION < 13000
CUresult CUDAAPI cuGraphAddNode(CUgraphNode *out, CUgraph g, const CUgraphNode *deps, size_t nd,
                                CUgraphNodeParams *p) {
  return legacyGraphAddNode(out, g, deps, nd, p);
}
#endif

// ---- Node queries and updates ----

CUresult CUDAAPI cuGraphNodeGetType(CUgraphNode n, CUgraphNodeType *t) {
  *t = n->type;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphGetNodes(CUgraph g, CUgraphNode *nodes, size_t *n) {
  if (nodes)
    for (size_t i = 0; i < *n && i < g->n; i++) nodes[i] = g->nodes[i];
  else
    *n = g->n;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphGetRootNodes(CUgraph g, CUgraphNode *roots, size_t *n) {
  size_t k = 0;
  for (size_t i = 0; i < g->n; i++)
    if (g->nodes[i]->ndeps == 0) {
      if (roots && k < *n) roots[k] = g->nodes[i];
      k++;
    }
  if (!roots) *n = k;
  return CUDA_SUCCESS;
}
static CUresult nodeDeps(CUgraphNode node, CUgraphNode *deps, CUgraphEdgeData *edges, size_t *n) {
  if (deps) {
    for (size_t i = 0; i < *n && i < node->ndeps; i++) {
      deps[i] = node->deps[i];
      if (edges) memset(&edges[i], 0, sizeof edges[i]);
    }
    if (*n > node->ndeps) *n = node->ndeps;
  } else
    *n = node->ndeps;
  return CUDA_SUCCESS;
}
static CUresult nodeDependents(CUgraphNode node, CUgraphNode *out, CUgraphEdgeData *edges, size_t *n) {
  CUgraph g = node->owner;
  size_t k = 0;
  for (size_t i = 0; i < g->n; i++)
    for (size_t j = 0; j < g->nodes[i]->ndeps; j++)
      if (g->nodes[i]->deps[j] == node) {
        if (out && k < *n) {
          out[k] = g->nodes[i];
          if (edges) memset(&edges[k], 0, sizeof edges[k]);
        }
        k++;
      }
  if (!out || k < *n) *n = k;
  return CUDA_SUCCESS;
}
static CUresult graphEdges(CUgraph g, CUgraphNode *from, CUgraphNode *to, CUgraphEdgeData *edges, size_t *n) {
  size_t k = 0;
  for (size_t i = 0; i < g->n; i++)
    for (size_t j = 0; j < g->nodes[i]->ndeps; j++) {
      if (from && to && k < *n) {
        from[k] = g->nodes[i]->deps[j];
        to[k] = g->nodes[i];
        if (edges) memset(&edges[k], 0, sizeof edges[k]);
      }
      k++;
    }
  if (!from || k < *n) *n = k;
  return CUDA_SUCCESS;
}
static CUresult addDeps(const CUgraphNode *from, const CUgraphNode *to, size_t n) {
  for (size_t i = 0; i < n; i++) addDep((CUgraphNode)to[i], (CUgraphNode)from[i]);
  return CUDA_SUCCESS;
}
static CUresult removeDeps(const CUgraphNode *from, const CUgraphNode *to, size_t n) {
  for (size_t i = 0; i < n; i++) {
    CUgraphNode t = to[i];
    for (size_t j = 0; j < t->ndeps; j++)
      if (t->deps[j] == from[i]) {
        t->deps[j] = t->deps[--t->ndeps];
        break;
      }
  }
  return CUDA_SUCCESS;
}

// Edge data (12.3 on) only orders programmatic launches, which run in order here: it is accepted
// and reported as the default edge.
CUresult CUDAAPI cuGraphNodeGetDependencies_v2(CUgraphNode node, CUgraphNode *deps, CUgraphEdgeData *edges, size_t *n) {
  return nodeDeps(node, deps, edges, n);
}
CUresult CUDAAPI cuGraphNodeGetDependentNodes_v2(CUgraphNode node, CUgraphNode *out, CUgraphEdgeData *edges, size_t *n) {
  return nodeDependents(node, out, edges, n);
}
CUresult CUDAAPI cuGraphGetEdges_v2(CUgraph g, CUgraphNode *from, CUgraphNode *to, CUgraphEdgeData *edges, size_t *n) {
  return graphEdges(g, from, to, edges, n);
}
CUresult CUDAAPI cuGraphAddDependencies_v2(CUgraph g, const CUgraphNode *from, const CUgraphNode *to,
                                           const CUgraphEdgeData *edges, size_t n) {
  (void)g; (void)edges;
  return addDeps(from, to, n);
}
CUresult CUDAAPI cuGraphRemoveDependencies_v2(CUgraph g, const CUgraphNode *from, const CUgraphNode *to,
                                              const CUgraphEdgeData *edges, size_t n) {
  (void)g; (void)edges;
  return removeDeps(from, to, n);
}
// The forms without edge data: CUDA 12's names (CUDA 13's headers give the names to the forms
// above; a CUDA 12 runtime still gets these, `legacy` in cuda.c).
static CUresult legacyNodeGetDependencies(CUgraphNode node, CUgraphNode *deps, size_t *n) { return nodeDeps(node, deps, NULL, n); }
static CUresult legacyNodeGetDependentNodes(CUgraphNode node, CUgraphNode *out, size_t *n) { return nodeDependents(node, out, NULL, n); }
static CUresult legacyGraphGetEdges(CUgraph g, CUgraphNode *from, CUgraphNode *to, size_t *n) { return graphEdges(g, from, to, NULL, n); }
static CUresult legacyAddDependencies(CUgraph g, const CUgraphNode *from, const CUgraphNode *to, size_t n) {
  (void)g;
  return addDeps(from, to, n);
}
static CUresult legacyRemoveDependencies(CUgraph g, const CUgraphNode *from, const CUgraphNode *to, size_t n) {
  (void)g;
  return removeDeps(from, to, n);
}
#if CUDA_VERSION < 13000
CUresult CUDAAPI cuGraphNodeGetDependencies(CUgraphNode node, CUgraphNode *deps, size_t *n) { return legacyNodeGetDependencies(node, deps, n); }
CUresult CUDAAPI cuGraphNodeGetDependentNodes(CUgraphNode node, CUgraphNode *out, size_t *n) { return legacyNodeGetDependentNodes(node, out, n); }
CUresult CUDAAPI cuGraphGetEdges(CUgraph g, CUgraphNode *from, CUgraphNode *to, size_t *n) { return legacyGraphGetEdges(g, from, to, n); }
CUresult CUDAAPI cuGraphAddDependencies(CUgraph g, const CUgraphNode *from, const CUgraphNode *to, size_t n) { return legacyAddDependencies(g, from, to, n); }
CUresult CUDAAPI cuGraphRemoveDependencies(CUgraph g, const CUgraphNode *from, const CUgraphNode *to, size_t n) { return legacyRemoveDependencies(g, from, to, n); }
#endif
CUresult CUDAAPI cuGraphChildGraphNodeGetGraph(CUgraphNode n, CUgraph *g) {
  *g = n->child;
  return n->child ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode n, CUDA_KERNEL_NODE_PARAMS_v2 *p) {
  if (n->type != CU_GRAPH_NODE_TYPE_KERNEL) return CUDA_ERROR_INVALID_VALUE;
  *p = n->kparams;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphKernelNodeSetParams(CUgraphNode n, const CUDA_KERNEL_NODE_PARAMS_v2 *p) {
  if (n->type != CU_GRAPH_NODE_TYPE_KERNEL) return CUDA_ERROR_INVALID_VALUE;
  Op *o;
  CUresult r = kernelNodeOp(&o, p, n->cluster);
  if (r != CUDA_SUCCESS) return r;
  opFree(n->op);
  n->op = o;
  n->kparams = *p;
  n->kparams.kernelParams = NULL;
  n->kparams.extra = NULL;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphKernelNodeSetAttribute(CUgraphNode n, CUkernelNodeAttrID a, const CUkernelNodeAttrValue *v) {
  if (n->type != CU_GRAPH_NODE_TYPE_KERNEL) return CUDA_ERROR_INVALID_VALUE;
  if (a == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
    n->cluster[0] = n->op->cluster[0] = v->clusterDim.x;
    n->cluster[1] = n->op->cluster[1] = v->clusterDim.y;
    n->cluster[2] = n->op->cluster[2] = v->clusterDim.z;
  }
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphKernelNodeGetAttribute(CUgraphNode n, CUkernelNodeAttrID a, CUkernelNodeAttrValue *v) {
  memset(v, 0, sizeof *v);
  if (a == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
    v->clusterDim.x = n->cluster[0];
    v->clusterDim.y = n->cluster[1];
    v->clusterDim.z = n->cluster[2];
  }
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
  memcpy(dst->cluster, src->cluster, sizeof dst->cluster);
  if (dst->op) memcpy(dst->op->cluster, src->cluster, sizeof dst->cluster);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphMemAllocNodeGetParams(CUgraphNode n, CUDA_MEM_ALLOC_NODE_PARAMS *p) {
  memset(p, 0, sizeof *p);
  p->bytesize = n->allocBytes;
  p->dptr = n->allocPtr;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphDestroyNode(CUgraphNode n) {
  CUgraph g = n->owner;
  for (size_t i = 0; i < g->n; i++) {
    CUgraphNode m = g->nodes[i];
    for (size_t j = 0; j < m->ndeps; j++)
      if (m->deps[j] == n) m->deps[j--] = m->deps[--m->ndeps];
  }
  size_t k = 0;
  for (size_t i = 0; i < g->n; i++)
    if (g->nodes[i] != n) {
      g->nodes[i]->order = k;
      g->nodes[k++] = g->nodes[i];
    }
  g->n = k;
  if (n->op) opFree(n->op);
  graphMemoryRelease(n->memory);
  if (n->child) graphFree(n->child);
  free(n->deps);
  free(n);
  return CUDA_SUCCESS;
}

// ---- Instantiation and launch ----

// The executable graph is a copy of the graph (`graphClone`; each node knows its original). It
// runs each graph's nodes in an order that respects every dependency (Kahn's order, ties in
// creation order), child graphs in place.
struct CUgraphExec_st {
  CUgraph graph;
  Completion *last;  // every launch follows the preceding one, including on other streams
  int refs;  // the handle and every launch of it not yet run (atomic)
};

static void execRetain(CUgraphExec e) { __atomic_add_fetch(&e->refs, 1, __ATOMIC_RELAXED); }
static void execRelease(CUgraphExec e) {
  if (__atomic_sub_fetch(&e->refs, 1, __ATOMIC_ACQ_REL) == 0) {
    graphFree(e->graph);
    completionRelease(e->last);
    free(e);
  }
}

// `g`'s nodes in an order that respects every dependency, into `out` (g->n entries); the count
// ordered (less than g->n where there is a cycle).
static size_t ordered(CUgraph g, CUgraphNode *out) {
  size_t m = g->n, k = 0;
  size_t *indeg = calloc(m ? m : 1, sizeof *indeg);
  for (size_t i = 0; i < m; i++) indeg[i] = g->nodes[i]->ndeps;
  // Each node's dependents, by index.
  size_t *cnt = calloc(m + 1, sizeof *cnt), *adj;
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < g->nodes[i]->ndeps; j++) cnt[g->nodes[i]->deps[j]->order + 1]++;
  for (size_t i = 0; i < m; i++) cnt[i + 1] += cnt[i];
  adj = malloc(sizeof *adj * (cnt[m] ? cnt[m] : 1));
  size_t *fill = calloc(m ? m : 1, sizeof *fill);
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < g->nodes[i]->ndeps; j++) {
      size_t d = g->nodes[i]->deps[j]->order;
      adj[cnt[d] + fill[d]++] = i;
    }
  // Ready nodes in creation order: a scan from the lowest ready index.
  int *done = calloc(m ? m : 1, sizeof *done);
  for (;;) {
    size_t pick = m;
    for (size_t i = 0; i < m; i++)
      if (!done[i] && indeg[i] == 0) {
        pick = i;
        break;
      }
    if (pick == m) break;
    done[pick] = 1;
    out[k++] = g->nodes[pick];
    for (size_t a = cnt[pick]; a < cnt[pick + 1]; a++) indeg[adj[a]]--;
  }
  free(indeg);
  free(cnt);
  free(adj);
  free(fill);
  free(done);
  return k;
}

static int acyclic(CUgraph g) {
  CUgraphNode *o = malloc(sizeof *o * (g->n ? g->n : 1));
  int ok = ordered(g, o) == g->n;
  for (size_t i = 0; ok && i < g->n; i++) {
    if (g->nodes[i]->child) ok = acyclic(g->nodes[i]->child);
    for (unsigned b = 0; ok && b < g->nodes[i]->nbodies; b++) ok = acyclic(g->nodes[i]->bodies[b]);
  }
  free(o);
  return ok;
}

CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec *out, CUgraph g, unsigned long long flags) {
  (void)flags;
  if (!acyclic(g)) return CUDA_ERROR_INVALID_VALUE;
  CUgraphExec e = calloc(1, sizeof *e);
  e->refs = 1;
  e->graph = graphClone(g);
  *out = e;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphInstantiateWithParams(CUgraphExec *out, CUgraph g, CUDA_GRAPH_INSTANTIATE_PARAMS *p) {
  CUresult r = cuGraphInstantiateWithFlags(out, g, p ? p->flags : 0);
  if (p) {
    p->hErrNode_out = NULL;
    p->result_out = r == CUDA_SUCCESS ? CUDA_GRAPH_INSTANTIATE_SUCCESS : CUDA_GRAPH_INSTANTIATE_ERROR;
  }
  return r;
}
// Launches still queued keep the executable until they run.
CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec e) {
  execRelease(e);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphUpload(CUgraphExec e, CUstream s) {
  (void)e; (void)s;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphExecGetFlags(CUgraphExec e, cuuint64_t *f) {
  (void)e;
  *f = 0;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDeviceGraphMemTrim(CUdevice d) {
  (void)d;
  return CUDA_SUCCESS;
}

// Run `g`'s work in order (from the stream the graph was launched on). A recording node numbers
// its recording as it runs; a wait node waits for the recording made before the launch.
static CUresult runGraph(CUgraph g);

// A conditional node: its handle's value read when it runs (and, for a while, after each run of
// its body).
static CUresult runConditional(CUgraphNode n) {
  for (;;) {
    unsigned v = 0;
    CUresult r = devRead(&v, n->cond, 4);
    if (r != CUDA_SUCCESS) return r;
    CUgraph body = NULL;
    if (n->condType == CU_GRAPH_COND_TYPE_WHILE || n->condType == CU_GRAPH_COND_TYPE_IF)
      body = v ? n->bodies[0] : n->nbodies > 1 ? n->bodies[1] : NULL;
    else if (v < n->nbodies)  // a switch
      body = n->bodies[v];
    if (!body) return CUDA_SUCCESS;
    r = runGraph(body);
    if (r != CUDA_SUCCESS || n->condType != CU_GRAPH_COND_TYPE_WHILE || !v) return r;
  }
}

// The handles of `g` and its children made with CU_GRAPH_COND_ASSIGN_DEFAULT reset.
static CUresult resetConds(CUgraph g) {
  CUresult r = CUDA_SUCCESS;
  for (size_t i = 0; i < g->nconds && r == CUDA_SUCCESS; i++)
    if (g->conds[i].assign) r = devWrite(g->conds[i].word, &g->conds[i].dflt, 4);
  for (size_t i = 0; i < g->n && r == CUDA_SUCCESS; i++) {
    if (g->nodes[i]->child) r = resetConds(g->nodes[i]->child);
    for (unsigned b = 0; b < g->nodes[i]->nbodies && r == CUDA_SUCCESS; b++) r = resetConds(g->nodes[i]->bodies[b]);
  }
  return r;
}

static CUresult runGraph(CUgraph g) {
  CUgraphNode *o = malloc(sizeof *o * (g->n ? g->n : 1));
  size_t k = ordered(g, o);
  CUresult r = CUDA_SUCCESS;
  for (size_t i = 0; i < k && r == CUDA_SUCCESS; i++) {
    CUgraphNode n = o[i];
    if (!n->enabled) continue;
    if (n->child) {
      r = runGraph(n->child);
      continue;
    }
    if (n->nbodies) {
      r = runConditional(n);
      continue;
    }
    Op *op = n->op;
    if (!op) continue;
    r = runOp(op);
    if (op->kind == OP_RECORD) finished(CUDA_SUCCESS);
  }
  free(o);
  return r;
}
static CUresult runExec(Op *o) {
  CUresult r = resetConds(o->graph);
  return r == CUDA_SUCCESS ? runGraph(o->graph) : r;
}

// Publish launch-specific event recordings at submission, and bind waits to the recording
// visible then. Child graphs share the parent's submission; conditional bodies cannot contain
// event nodes (CUDA's conditional-node contract).
static void armEvents(CUgraph g) {
  CUgraphNode *order = malloc(sizeof *order * (g->n ? g->n : 1));
  size_t count = ordered(g, order);
  for (size_t i = 0; i < count; i++) {
    CUgraphNode n = order[i];
    if (!n->enabled) continue;
    if (n->child) armEvents(n->child);
    if (n->op && n->op->kind == OP_WAIT) n->op->seq = n->op->ev->recorded;
    if (n->op && n->op->kind == OP_RECORD) n->op->seq = ++n->op->ev->recorded;
  }
  free(order);
}
static void armLaunch(Op *o) {
  dependsOn(o, o->exec->last);
  completionReplace(&o->exec->last, o->completion);
  armEvents(o->graph);
}
CUresult CUDAAPI cuGraphLaunch(CUgraphExec e, CUstream s) {
  NEED_INIT();
  Op *o = newOp(OP_GRAPH);
  execRetain(e);
  o->exec = e;
  o->graph = graphClone(e->graph);
  return submit(s, o);
}

// The executable's copy of node `n`: the copy made from it, or, for a node of a graph the
// executable was updated from (whose topology is the instantiated one's), the copy at its place.
static CUgraphNode findIn(CUgraph g, CUgraphNode n) {
  for (size_t i = 0; i < g->n; i++) {
    CUgraphNode m = g->nodes[i];
    if (m->origin == n || m->root == n) return m;
    if (m->child) {
      CUgraphNode c = findIn(m->child, n);
      if (c) return c;
    }
    for (unsigned b = 0; b < m->nbodies; b++) {
      CUgraphNode c = findIn(m->bodies[b], n);
      if (c) return c;
    }
  }
  return NULL;
}
static CUgraphNode execNode(CUgraphExec e, CUgraphNode n) {
  CUgraphNode m = findIn(e->graph, n);
  if (!m && n->owner && n->owner->n == e->graph->n && n->order < e->graph->n &&
      e->graph->nodes[n->order]->type == n->type)
    m = e->graph->nodes[n->order];
  return m;
}

// Node `m`'s work replaced by `o` (it owns its work).
static void replaceOp(CUgraphNode m, Op *o) {
  if (m->op) opFree(m->op);
  m->op = o;
}

CUresult CUDAAPI cuGraphExecKernelNodeSetParams(CUgraphExec e, CUgraphNode n, const CUDA_KERNEL_NODE_PARAMS_v2 *p) {
  CUgraphNode m = execNode(e, n);
  if (!m || m->type != CU_GRAPH_NODE_TYPE_KERNEL) return CUDA_ERROR_INVALID_VALUE;
  Op *o;
  CUresult r = kernelNodeOp(&o, p, m->cluster);
  if (r == CUDA_SUCCESS) replaceOp(m, o);
  return r;
}
CUresult CUDAAPI cuGraphExecMemcpyNodeSetParams(CUgraphExec e, CUgraphNode n, const CUDA_MEMCPY3D *c, CUcontext ctx) {
  (void)ctx;
  CUgraphNode m = execNode(e, n);
  Op *o = m && m->type == CU_GRAPH_NODE_TYPE_MEMCPY ? copy3DOp(c) : NULL;
  if (!o) return CUDA_ERROR_INVALID_VALUE;
  replaceOp(m, o);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphExecMemsetNodeSetParams(CUgraphExec e, CUgraphNode n, const CUDA_MEMSET_NODE_PARAMS *p, CUcontext ctx) {
  (void)ctx;
  CUgraphNode m = execNode(e, n);
  if (!m || m->type != CU_GRAPH_NODE_TYPE_MEMSET) return CUDA_ERROR_INVALID_VALUE;
  replaceOp(m, memsetNodeOp(p->dst, p->pitch, p->value, p->elementSize, p->width, p->height));
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphExecBatchMemOpNodeSetParams(CUgraphExec e, CUgraphNode n,
                                                    const CUDA_BATCH_MEM_OP_NODE_PARAMS *p) {
  CUgraphNode m = execNode(e, n);
  return m ? cuGraphBatchMemOpNodeSetParams(m, p) : CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI cuGraphExecHostNodeSetParams(CUgraphExec e, CUgraphNode n, const CUDA_HOST_NODE_PARAMS *p) {
  CUgraphNode m = execNode(e, n);
  return m ? cuGraphHostNodeSetParams(m, p) : CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI cuGraphExecChildGraphNodeSetParams(CUgraphExec e, CUgraphNode n, CUgraph child) {
  CUgraphNode m = execNode(e, n);
  if (!m || m->type != CU_GRAPH_NODE_TYPE_GRAPH) return CUDA_ERROR_INVALID_VALUE;
  graphFree(m->child);
  m->child = graphClone(child);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphExecEventRecordNodeSetEvent(CUgraphExec e, CUgraphNode n, CUevent ev) {
  CUgraphNode m = execNode(e, n);
  if (!m || m->type != CU_GRAPH_NODE_TYPE_EVENT_RECORD) return CUDA_ERROR_INVALID_VALUE;
  replaceOp(m, eventOp(OP_RECORD, ev));
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphExecEventWaitNodeSetEvent(CUgraphExec e, CUgraphNode n, CUevent ev) {
  CUgraphNode m = execNode(e, n);
  if (!m || m->type != CU_GRAPH_NODE_TYPE_WAIT_EVENT) return CUDA_ERROR_INVALID_VALUE;
  replaceOp(m, eventOp(OP_WAIT, ev));
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphNodeSetEnabled(CUgraphExec e, CUgraphNode n, unsigned int on) {
  CUgraphNode m = execNode(e, n);
  if (!m) return CUDA_ERROR_INVALID_VALUE;
  m->enabled = (int)on;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuGraphNodeGetEnabled(CUgraphExec e, CUgraphNode n, unsigned int *on) {
  CUgraphNode m = execNode(e, n);
  if (!m) return CUDA_ERROR_INVALID_VALUE;
  *on = (unsigned)m->enabled;
  return CUDA_SUCCESS;
}

// A node's parameters set from the generic form (cuGraphNodeSetParams on a graph's node,
// cuGraphExecNodeSetParams on an executable's copy).
static CUresult setNodeParams(CUgraphNode m, CUgraphNodeParams *p) {
  if (p->type != m->type) return CUDA_ERROR_INVALID_VALUE;
  switch (p->type) {
    case CU_GRAPH_NODE_TYPE_KERNEL: {
      CUDA_KERNEL_NODE_PARAMS_v2 k;
      memcpy(&k, &p->kernel, sizeof k);
      Op *o;
      CUresult r = kernelNodeOp(&o, &k, m->cluster);
      if (r != CUDA_SUCCESS) return r;
      replaceOp(m, o);
      m->kparams = k;
      m->kparams.kernelParams = NULL;
      m->kparams.extra = NULL;
      return CUDA_SUCCESS;
    }
    case CU_GRAPH_NODE_TYPE_MEMCPY: {
      Op *o = copy3DOp(&p->memcpy.copyParams);
      if (!o) return CUDA_ERROR_NOT_SUPPORTED;
      replaceOp(m, o);
      return CUDA_SUCCESS;
    }
    case CU_GRAPH_NODE_TYPE_MEMSET:
      replaceOp(m, memsetNodeOp(p->memset.dst, p->memset.pitch, p->memset.value, p->memset.elementSize,
                                p->memset.width, p->memset.height));
      return CUDA_SUCCESS;
    case CU_GRAPH_NODE_TYPE_HOST: {
      CUDA_HOST_NODE_PARAMS h = {p->host.fn, p->host.userData};
      return cuGraphHostNodeSetParams(m, &h);
    }
    case CU_GRAPH_NODE_TYPE_GRAPH:
      graphFree(m->child);
      m->child = graphClone(p->graph.graph);
      return CUDA_SUCCESS;
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD: replaceOp(m, eventOp(OP_RECORD, p->eventRecord.event)); return CUDA_SUCCESS;
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT: replaceOp(m, eventOp(OP_WAIT, p->eventWait.event)); return CUDA_SUCCESS;
    case CU_GRAPH_NODE_TYPE_EMPTY: return CUDA_SUCCESS;
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: {
      CUDA_BATCH_MEM_OP_NODE_PARAMS b = {.ctx = p->memOp.ctx, .count = p->memOp.count,
                                         .paramArray = p->memOp.paramArray, .flags = p->memOp.flags};
      return cuGraphBatchMemOpNodeSetParams(m, &b);
    }
    default:
      note("setting parameters of graph node type %d is not modelled", (int)p->type);
      return CUDA_ERROR_NOT_SUPPORTED;
  }
}
CUresult CUDAAPI cuGraphNodeSetParams(CUgraphNode n, CUgraphNodeParams *p) { return setNodeParams(n, p); }
CUresult CUDAAPI cuGraphExecNodeSetParams(CUgraphExec e, CUgraphNode n, CUgraphNodeParams *p) {
  CUgraphNode m = execNode(e, n);
  return m ? setNodeParams(m, p) : CUDA_ERROR_INVALID_VALUE;
}

// An update gives the executable the graph's work, its topology the instantiated one's (the
// driver's rule): each node's work copied into the executable's node at the same place.
static int sameTopology(CUgraph a, CUgraph b) {
  if (a->n != b->n) return 0;
  for (size_t i = 0; i < a->n; i++) {
    CUgraphNode x = a->nodes[i], y = b->nodes[i];
    if (x->type != y->type || x->ndeps != y->ndeps) return 0;
    for (size_t j = 0; j < x->ndeps; j++)
      if (x->deps[j]->order != y->deps[j]->order) return 0;
  }
  return 1;
}
static void updateFrom(CUgraph exec, CUgraph g) {
  for (size_t i = 0; i < g->n; i++) {
    CUgraphNode m = exec->nodes[i], n = g->nodes[i];
    if (n->child) {
      if (m->child && sameTopology(m->child, n->child))
        updateFrom(m->child, n->child);
      else {
        if (m->child) graphFree(m->child);
        m->child = graphClone(n->child);
      }
    } else if (n->nbodies) {
      for (unsigned b = 0; b < n->nbodies && b < m->nbodies; b++) {
        graphFree(m->bodies[b]);
        m->bodies[b] = graphClone(n->bodies[b]);
      }
      m->cond = n->cond;
      m->condType = n->condType;
    } else if (n->op)
      replaceOp(m, opCopy(n->op));
    m->kparams = n->kparams;
    memcpy(m->cluster, n->cluster, sizeof m->cluster);
    m->enabled = n->enabled;
  }
  graphConditions(exec, g);
}
CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec e, CUgraph g, CUgraphExecUpdateResultInfo *info) {
  if (info) memset(info, 0, sizeof *info);
  if (!sameTopology(e->graph, g)) {
    if (info) info->result = CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED;
    return CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE;
  }
  updateFrom(e->graph, g);
  if (info) info->result = CU_GRAPH_EXEC_UPDATE_SUCCESS;
  return CUDA_SUCCESS;
}

// ---- Stream capture ----

static unsigned long long captureIds;

static void setDeps(CUstream s, CUgraphNode *deps, size_t n) {
  if (n > s->depsCap) s->deps = realloc(s->deps, sizeof *s->deps * (s->depsCap = n));
  memcpy(s->deps, deps, sizeof *deps * n);
  s->ndeps = n;
}

// (workLock held) `o` as a node of the stream's capture, after the stream's last nodes.
static CUresult captureOp(CUstream s, Op *o) {
  CUgraph g = s->capture;
  if (o->kind == OP_RECORD && !(o->flags & CU_EVENT_RECORD_EXTERNAL)) {
    // The event carries the nodes before it; nothing is added to the graph.
    CUevent e = o->ev;
    free(e->nodes);
    e->nodes = malloc(sizeof *e->nodes * (s->ndeps ? s->ndeps : 1));
    memcpy(e->nodes, s->deps, sizeof *e->nodes * s->ndeps);
    e->nnodes = s->ndeps;
    e->capture = g;
    e->captureId = s->captureId;
    o->ev = NULL;
    eventRelease(e);
    free(o->params);
    free(o);
    return CUDA_SUCCESS;
  }
  if (o->kind == OP_WAIT) {
    CUevent e = o->ev;
    if (!(o->flags & CU_EVENT_WAIT_EXTERNAL) && e->capture == g && e->captureId == s->captureId) {
      for (size_t i = 0; i < e->nnodes; i++) {
        int seen = 0;
        for (size_t j = 0; j < s->ndeps; j++) seen |= s->deps[j] == e->nodes[i];
        if (!seen) {
          if (s->ndeps == s->depsCap) s->deps = realloc(s->deps, sizeof *s->deps * (s->depsCap = s->depsCap ? 2 * s->depsCap : 4));
          s->deps[s->ndeps++] = e->nodes[i];
        }
      }
      o->ev = NULL;
      eventRelease(e);
      free(o);
      return CUDA_SUCCESS;
    }
    // A wait on an event recorded outside the capture becomes a wait node.
  }
  CUgraphNodeType t = o->kind == OP_KERNEL ? CU_GRAPH_NODE_TYPE_KERNEL
                    : o->kind == OP_COPY ? CU_GRAPH_NODE_TYPE_MEMCPY
                    : o->kind == OP_SET ? CU_GRAPH_NODE_TYPE_MEMSET
                    : o->kind == OP_HOST || o->kind == OP_CALLBACK ? CU_GRAPH_NODE_TYPE_HOST
                    : o->kind == OP_RECORD ? CU_GRAPH_NODE_TYPE_EVENT_RECORD
                    : o->kind == OP_FREE ? CU_GRAPH_NODE_TYPE_MEM_FREE
                    : o->kind == OP_WAIT ? CU_GRAPH_NODE_TYPE_WAIT_EVENT
                    : o->kind == OP_GRAPH ? CU_GRAPH_NODE_TYPE_GRAPH
                    : o->kind == OP_MEMOPS ? CU_GRAPH_NODE_TYPE_BATCH_MEM_OP
                    : CU_GRAPH_NODE_TYPE_EMPTY;
  CUgraphNode n;
  if (o->kind == OP_GRAPH) {
    // A graph launched in a capture: its executable's graph becomes a child.
    n = newNode(g, CU_GRAPH_NODE_TYPE_GRAPH, NULL, s->deps, s->ndeps);
    n->child = o->graph;
    o->graph = NULL;
    opFree(o);
  } else {
    n = newNode(g, t, o, s->deps, s->ndeps);
    if (o->kind == OP_KERNEL) {
      n->kparams.func = o->f;
      n->kparams.gridDimX = o->grid[0];
      n->kparams.gridDimY = o->grid[1];
      n->kparams.gridDimZ = o->grid[2];
      n->kparams.blockDimX = o->block[0];
      n->kparams.blockDimY = o->block[1];
      n->kparams.blockDimZ = o->block[2];
      n->kparams.sharedMemBytes = o->smem;
      memcpy(n->cluster, o->cluster, sizeof n->cluster);
    }
  }
  setDeps(s, &n, 1);
  return CUDA_SUCCESS;
}

static CUresult captureAlloc(CUstream s, CUdeviceptr *p, size_t bytes) {
  CUDA_MEM_ALLOC_NODE_PARAMS params = {0};
  params.bytesize = bytes;
  CUgraphNode n;
  CUresult r = cuGraphAddMemAllocNode(&n, s->capture, s->deps, s->ndeps, &params);
  if (r == CUDA_SUCCESS) {
    *p = params.dptr;
    setDeps(s, &n, 1);
  }
  return r;
}

static CUresult beginCapture(CUstream s, CUgraph g, CUstreamCaptureMode mode) {
  if (isLegacy(s)) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
  pthread_mutex_lock(&workLock);
  if (s->capture) {
    pthread_mutex_unlock(&workLock);
    return CUDA_ERROR_ILLEGAL_STATE;
  }
  s->capture = g;
  s->captureOrigin = s;
  s->captureOwnsGraph = 1;
  s->captureBroken = 0;
  s->captureMode = mode;
  s->captureId = ++captureIds;
  s->ndeps = 0;
  pthread_mutex_unlock(&workLock);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamBeginCapture(CUstream s, CUstreamCaptureMode mode) {
  CUgraph g;
  cuGraphCreate(&g, 0);
  CUresult r = beginCapture(s, g, mode);
  if (r != CUDA_SUCCESS) graphFree(g);
  return r;
}
#if CUDA_VERSION >= 12030
CUresult CUDAAPI cuStreamBeginCaptureToGraph(CUstream s, CUgraph g, const CUgraphNode *deps,
                                             const CUgraphEdgeData *edges, size_t n,
                                             CUstreamCaptureMode mode) {
  (void)edges;
  CUresult r = beginCapture(s, g, mode);
  if (r == CUDA_SUCCESS) {
    pthread_mutex_lock(&workLock);
    setDeps(s, (CUgraphNode *)deps, n);
    s->captureOwnsGraph = 0;
    pthread_mutex_unlock(&workLock);
  }
  return r;
}
#endif

// Mark the ancestors of the origin's final frontier. Every joined branch must finish there.
static void markAncestors(CUgraphNode n, unsigned char *seen) {
  if (seen[n->order]) return;
  seen[n->order] = 1;
  for (size_t i = 0; i < n->ndeps; i++) markAncestors(n->deps[i], seen);
}
CUresult CUDAAPI cuStreamEndCapture(CUstream s, CUgraph *out) {
  if (isLegacy(s)) return CUDA_ERROR_ILLEGAL_STATE;
  pthread_mutex_lock(&workLock);
  CUgraph g = s->capture;
  if (!g || s->captureOrigin != s) {
    pthread_mutex_unlock(&workLock);
    return g ? CUDA_ERROR_STREAM_CAPTURE_UNMATCHED : CUDA_ERROR_ILLEGAL_STATE;
  }
  unsigned char *seen = calloc(g->n ? g->n : 1, 1);
  for (size_t i = 0; i < s->ndeps; i++) markAncestors(s->deps[i], seen);
  CUresult r = s->captureBroken ? CUDA_ERROR_STREAM_CAPTURE_INVALIDATED : CUDA_SUCCESS;
  for (CUstream t = streams; t; t = t->nextStream)
    if (t->capture == g) {
      for (size_t i = 0; !r && i < t->ndeps; i++)
        if (!seen[t->deps[i]->order]) r = CUDA_ERROR_STREAM_CAPTURE_UNJOINED;
      t->capture = NULL;
      t->captureOrigin = NULL;
      t->ndeps = 0;
      t->captureBroken = 0;
    }
  free(seen);
  int owns = s->captureOwnsGraph;
  pthread_mutex_unlock(&workLock);
  if (r) {
    if (owns) graphFree(g);
    *out = NULL;
    return r;
  }
  *out = g;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuStreamIsCapturing(CUstream s, CUstreamCaptureStatus *st) {
  pthread_mutex_lock(&workLock);
  *st = !isLegacy(s) && s->capture ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
  pthread_mutex_unlock(&workLock);
  return CUDA_SUCCESS;
}

static CUresult captureInfo(CUstream s, CUstreamCaptureStatus *st, cuuint64_t *id, CUgraph *g,
                            const CUgraphNode **deps, size_t *n) {
  pthread_mutex_lock(&workLock);
  int on = !isLegacy(s) && s->capture;
  if (st) *st = on ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
  if (id) *id = on ? s->captureId : 0;
  if (g) *g = on ? s->capture : NULL;
  if (deps) *deps = on ? s->deps : NULL;
  if (n) *n = on ? s->ndeps : 0;
  pthread_mutex_unlock(&workLock);
  return CUDA_SUCCESS;
}

static CUresult legacyStreamGetCaptureInfo(CUstream s, CUstreamCaptureStatus *st, cuuint64_t *id,
                                           CUgraph *g, const CUgraphNode **deps, size_t *n) {
  return captureInfo(s, st, id, g, deps, n);
}
// cudaTypedefs.h, PFN_cuStreamGetCaptureInfo_v10010: only status and id are outputs.
static CUresult legacyStreamGetCaptureInfo_v1(CUstream s, CUstreamCaptureStatus *st, cuuint64_t *id) {
  return captureInfo(s, st, id, NULL, NULL, NULL);
}
CUresult CUDAAPI cuStreamGetCaptureInfo_v3(CUstream s, CUstreamCaptureStatus *st, cuuint64_t *id,
                                           CUgraph *g, const CUgraphNode **deps,
                                           const CUgraphEdgeData **edges, size_t *n) {
  if (edges) *edges = NULL;  // every edge the default
  return captureInfo(s, st, id, g, deps, n);
}
#if CUDA_VERSION < 13000
CUresult CUDAAPI cuStreamGetCaptureInfo_v2(CUstream s, CUstreamCaptureStatus *st, cuuint64_t *id,
                                           CUgraph *g, const CUgraphNode **deps, size_t *n) {
  return captureInfo(s, st, id, g, deps, n);
}
#endif

static CUresult updateDeps(CUstream s, CUgraphNode *deps, size_t n, unsigned int flags) {
  pthread_mutex_lock(&workLock);
  if (isLegacy(s) || !s->capture) {
    pthread_mutex_unlock(&workLock);
    return CUDA_ERROR_ILLEGAL_STATE;
  }
  if (flags == CU_STREAM_SET_CAPTURE_DEPENDENCIES)
    setDeps(s, deps, n);
  else
    for (size_t i = 0; i < n; i++) {
      if (s->ndeps == s->depsCap) s->deps = realloc(s->deps, sizeof *s->deps * (s->depsCap = s->depsCap ? 2 * s->depsCap : 4));
      s->deps[s->ndeps++] = deps[i];
    }
  pthread_mutex_unlock(&workLock);
  return CUDA_SUCCESS;
}

static CUresult legacyStreamUpdateCaptureDependencies(CUstream s, CUgraphNode *deps, size_t n,
                                                      unsigned int flags) {
  return updateDeps(s, deps, n, flags);
}
CUresult CUDAAPI cuStreamUpdateCaptureDependencies_v2(CUstream s, CUgraphNode *deps,
                                                      const CUgraphEdgeData *edges, size_t n,
                                                      unsigned int flags) {
  (void)edges;
  return updateDeps(s, deps, n, flags);
}
#if CUDA_VERSION < 13000
CUresult CUDAAPI cuStreamUpdateCaptureDependencies(CUstream s, CUgraphNode *deps, size_t n,
                                                   unsigned int flags) {
  return updateDeps(s, deps, n, flags);
}
#endif

CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
  static __thread CUstreamCaptureMode m = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUstreamCaptureMode old = m;
  m = *mode;
  *mode = old;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGraphDebugDotPrint(CUgraph g, const char *path, unsigned int flags) {
  (void)flags;
  FILE *f = fopen(path, "w");
  if (!f) return CUDA_ERROR_OPERATING_SYSTEM;
  fprintf(f, "digraph G {\n");
  for (size_t i = 0; i < g->n; i++) {
    fprintf(f, "  n%zu [label=\"%zu: type %d\"];\n", i, i, (int)g->nodes[i]->type);
    for (size_t j = 0; j < g->nodes[i]->ndeps; j++) fprintf(f, "  n%zu -> n%zu;\n", g->nodes[i]->deps[j]->order, i);
  }
  fprintf(f, "}\n");
  fclose(f);
  return CUDA_SUCCESS;
}
