/* The emulated device's NCCL (libnccl.so.2): the collectives a mesh of emulated devices
(NVGPU_LEAN_GPU=b200x8) runs, for the programs that load NCCL to run them (XLA dlopens
libnccl.so.2). Its ABI is NCCL 2.32.6's (nccl.h; the nvidia-nccl-cu13 wheel's), its version that
one's.

Each collective is stream-ordered work: a host function on the caller's stream (cuLaunchHostFunc).
A communicator belongs to a clique, the ranks that share its unique id. The n ranks' calls of
their k-th collective meet in the clique: each waits there until all have arrived, the last
performs it, through the host (the device memory of every rank read, reduced, written), and all
go on. Reductions run in rank order (rank 0's values, then rank 1's, ...), each step in the data
type's arithmetic (16- and 8-bit floats through float, rounded to nearest even after each step),
so a result does not depend on timing; it is not NVIDIA NCCL's ring or tree order. A send is
complete once its data is staged, so a group of sends and receives between two ranks cannot wait
on each other; a receive waits for its send. Work captured into a CUDA graph (XLA's command
buffers) runs at each replay; it is numbered as it runs, so replays meet as calls do.

Ranks must run on streams of their own (not the legacy default stream of one thread), as XLA's
do: a collective waits on its stream's worker for the other ranks' calls. Window and device-side
APIs (symmetric memory, ncclDevComm) are not provided. */

#include <cuda.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NCCL_VERSION_CODE 23206
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;
typedef enum {
  ncclSuccess = 0, ncclUnhandledCudaError = 1, ncclSystemError = 2, ncclInternalError = 3,
  ncclInvalidArgument = 4, ncclInvalidUsage = 5, ncclRemoteError = 6, ncclInProgress = 7,
  ncclTimeout = 8
} ncclResult_t;
typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4, ncclNumOps = 5 } ncclRedOp_t;
typedef enum {
  ncclInt8 = 0, ncclUint8 = 1, ncclInt32 = 2, ncclUint32 = 3, ncclInt64 = 4, ncclUint64 = 5,
  ncclFloat16 = 6, ncclFloat32 = 7, ncclFloat64 = 8, ncclBfloat16 = 9, ncclFloat8e4m3 = 10,
  ncclFloat8e5m2 = 11, ncclNumTypes = 12
} ncclDataType_t;
typedef enum { ncclScalarDevice = 0, ncclScalarHostImmediate = 1 } ncclScalarResidence_t;
typedef void ncclConfig_t;  // read for nothing: blocking or not, a call here completes
typedef void ncclSimInfo_t;
typedef void *ncclWindow_t;

// The libcuda's device memory through the host (cuda.c), callable from a stream's host function.
CUresult nvgpuLeanRead(void *dst, CUdeviceptr src, size_t n);
CUresult nvgpuLeanWrite(CUdeviceptr dst, const void *src, size_t n);
uint64_t nvgpuLeanDeviceNs(int g);
void nvgpuLeanDeviceWaitNs(int g, uint64_t ns);
uint64_t nvgpuLeanCollectiveNs(int g, int kind, int ranks, uint64_t bytes);

// ---- Values: each type's bytes to double and back ----

static size_t typeBytes(ncclDataType_t t) {
  switch (t) {
    case ncclInt8: case ncclUint8: case ncclFloat8e4m3: case ncclFloat8e5m2: return 1;
    case ncclFloat16: case ncclBfloat16: return 2;
    case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
    case ncclInt64: case ncclUint64: case ncclFloat64: return 8;
    default: return 0;
  }
}
static int isFloat(ncclDataType_t t) {
  return t == ncclFloat16 || t == ncclFloat32 || t == ncclFloat64 || t == ncclBfloat16 ||
         t == ncclFloat8e4m3 || t == ncclFloat8e5m2;
}

// A float of `ebits` exponent and `mbits` mantissa bits (bias 2^(ebits-1) - 1) to float. e4m3fn
// has no infinities and one NaN pattern (all ones).
static float minifloatTo(uint32_t x, int ebits, int mbits, int fn) {
  uint32_t sign = x >> (ebits + mbits), e = (x >> mbits) & ((1u << ebits) - 1), m = x & ((1u << mbits) - 1);
  int bias = (1 << (ebits - 1)) - 1;
  float v;
  if (e == (1u << ebits) - 1 && (!fn || m == (1u << mbits) - 1))
    v = fn || m ? NAN : INFINITY;
  else if (e == 0) v = ldexpf((float)m, 1 - bias - mbits);
  else v = ldexpf((float)(m | (1u << mbits)), (int)e - bias - mbits);
  return sign ? -v : v;
}
// Float to such a float, rounded to nearest even; past the largest finite, infinity (or for e4m3fn
// the largest finite, as a saturating conversion's NaN-free result is not NCCL's; NaN stays NaN).
static uint32_t minifloatFrom(float f, int ebits, int mbits, int fn) {
  uint32_t sign = signbit(f) ? 1u << (ebits + mbits) : 0;
  int bias = (1 << (ebits - 1)) - 1;
  uint32_t maxE = (1u << ebits) - 1;
  if (isnan(f)) return sign | (maxE << mbits) | ((1u << mbits) - 1);
  float a = fabsf(f);
  // The largest finite: e4m3fn's is exponent all ones with mantissa one short of all ones.
  uint32_t maxFinite = fn ? (maxE << mbits) | ((1u << mbits) - 2) : ((maxE - 1) << mbits) | ((1u << mbits) - 1);
  if (isinf(a)) return fn ? sign | ((maxE << mbits) | ((1u << mbits) - 1)) : sign | (maxE << mbits);
  int ex;
  frexpf(a, &ex);  // a = f * 2^ex, f in [0.5, 1)
  int e = ex - 1 + bias;
  int shift = e <= 0 ? 1 - bias - mbits : ex - 1 - mbits;  // the unit in the last place, as a power
  float q = ldexpf(a, -shift);  // a in units of the last place
  float r = nearbyintf(q);      // ties to even (the default rounding mode)
  uint32_t bits = (uint32_t)r;
  uint32_t code;
  if (e <= 0) code = bits;  // subnormal (or rounded up to the least normal: same encoding rule)
  else {
    // bits in [2^mbits, 2^(mbits+1)]: the rounded significand, carrying into the exponent
    if (bits >> (mbits + 1)) { bits >>= 1; e++; }
    code = ((uint32_t)e << mbits) | (bits & ((1u << mbits) - 1));
  }
  if (code > maxFinite) return fn ? sign | ((maxE << mbits) | ((1u << mbits) - 1)) : sign | (maxE << mbits);
  return sign | code;
}
static float halfTo(uint16_t h) { return minifloatTo(h, 5, 10, 0); }
static uint16_t halfFrom(float f) { return (uint16_t)minifloatFrom(f, 5, 10, 0); }
static float bf16To(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint16_t bf16From(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if (isnan(f)) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fff + ((u >> 16) & 1);  // round to nearest even
  return (uint16_t)(u >> 16);
}

typedef struct {
  ncclRedOp_t base;  // ncclSum for a PreMulSum
  int premul;        // ncclRedOpCreatePreMulSum's: each input times `scalar` first
  double scalar;
  const void *deviceScalar;  // the scalar in device memory, read at the collective
  ncclDataType_t type;
  int used;
} RedOp;
#define MAX_OPS 64
static RedOp ops[MAX_OPS];
static pthread_mutex_t opLock = PTHREAD_MUTEX_INITIALIZER;

// Element i of `buf` as a double: floats, and integers for a PreMulSum's scale.
static double get(ncclDataType_t t, const void *buf, size_t i) {
  const unsigned char *b = buf;
  switch (t) {
    case ncclInt8: return ((const int8_t *)b)[i];
    case ncclUint8: return b[i];
    case ncclInt32: return ((const int32_t *)b)[i];
    case ncclUint32: return ((const uint32_t *)b)[i];
    case ncclFloat16: return halfTo(((const uint16_t *)b)[i]);
    case ncclBfloat16: return bf16To(((const uint16_t *)b)[i]);
    case ncclFloat32: return ((const float *)b)[i];
    case ncclFloat64: return ((const double *)b)[i];
    case ncclFloat8e4m3: return minifloatTo(b[i], 4, 3, 1);
    case ncclFloat8e5m2: return minifloatTo(b[i], 5, 2, 0);
    case ncclInt64: return (double)((const int64_t *)b)[i];
    case ncclUint64: return (double)((const uint64_t *)b)[i];
    default: return 0;
  }
}
// Stores v, rounded to the type (16- and 8-bit floats through float: v is a float's exact
// operation, so rounding it first to float and then to the type is rounding it once).
static void put(ncclDataType_t t, void *buf, size_t i, double v) {
  unsigned char *b = buf;
  switch (t) {
    case ncclInt8: ((int8_t *)b)[i] = (int8_t)(int64_t)v; break;
    case ncclUint8: b[i] = (uint8_t)(int64_t)v; break;
    case ncclInt32: ((int32_t *)b)[i] = (int32_t)(int64_t)v; break;
    case ncclUint32: ((uint32_t *)b)[i] = (uint32_t)(int64_t)v; break;
    case ncclFloat16: ((uint16_t *)b)[i] = halfFrom((float)v); break;
    case ncclBfloat16: ((uint16_t *)b)[i] = bf16From((float)v); break;
    case ncclFloat32: ((float *)b)[i] = (float)v; break;
    case ncclFloat64: ((double *)b)[i] = v; break;
    case ncclFloat8e4m3: b[i] = (uint8_t)minifloatFrom((float)v, 4, 3, 1); break;
    case ncclFloat8e5m2: b[i] = (uint8_t)minifloatFrom((float)v, 5, 2, 0); break;
    case ncclInt64: ((int64_t *)b)[i] = (int64_t)v; break;
    case ncclUint64: ((uint64_t *)b)[i] = (uint64_t)v; break;
    default: break;
  }
}

// An integer element of `buf` as a 64-bit integer (unsigned ones zero-extended).
static int64_t getInt(ncclDataType_t t, const void *buf, size_t i) {
  switch (t) {
    case ncclInt8: return ((const int8_t *)buf)[i];
    case ncclUint8: return ((const uint8_t *)buf)[i];
    case ncclInt32: return ((const int32_t *)buf)[i];
    case ncclUint32: return ((const uint32_t *)buf)[i];
    default: return ((const int64_t *)buf)[i];
  }
}
static void putInt(ncclDataType_t t, void *buf, size_t i, uint64_t v) {
  switch (t) {
    case ncclInt8: case ncclUint8: ((uint8_t *)buf)[i] = (uint8_t)v; break;
    case ncclInt32: case ncclUint32: ((uint32_t *)buf)[i] = (uint32_t)v; break;
    default: ((uint64_t *)buf)[i] = v; break;
  }
}
static int isSigned(ncclDataType_t t) { return t == ncclInt8 || t == ncclInt32 || t == ncclInt64; }

// acc[i] = acc[i] op x[i] in the type's arithmetic: a float's exact result rounded to its type
// (computed in double, exact for a sum or product of two floats of 24 bits or fewer, then rounded
// once, as float arithmetic rounds), integers wrapping as the hardware's do.
static void combine(ncclDataType_t t, ncclRedOp_t op, void *acc, const void *x, size_t n) {
  if (!isFloat(t)) {
    int s = isSigned(t);
    for (size_t i = 0; i < n; i++) {
      int64_t a = getInt(t, acc, i), b = getInt(t, x, i);
      uint64_t ua = (uint64_t)a, ub = (uint64_t)b, r;
      switch (op) {
        case ncclSum: case ncclAvg: r = ua + ub; break;
        case ncclProd: r = ua * ub; break;
        case ncclMax: r = s ? (uint64_t)(a > b ? a : b) : (ua > ub ? ua : ub); break;
        default: r = s ? (uint64_t)(a < b ? a : b) : (ua < ub ? ua : ub); break;
      }
      putInt(t, acc, i, r);
    }
    return;
  }
  for (size_t i = 0; i < n; i++) {
    double a = get(t, acc, i), b = get(t, x, i), r;
    switch (op) {
      case ncclSum: case ncclAvg: r = a + b; break;
      case ncclProd: r = a * b; break;
      case ncclMax: r = isnan(a) || a > b ? a : b; break;
      default: r = isnan(a) || a < b ? a : b; break;
    }
    put(t, acc, i, r);
  }
}

// ---- Cliques and communicators ----

typedef struct Task Task;
typedef struct Clique Clique;

struct ncclComm {
  Clique *clique;
  int rank, device;
  // Counted as its work runs, in its streams' order (a captured collective runs once a replay):
  uint64_t seq;                 // its collectives run so far
  uint64_t *sendSeq, *recvSeq;  // per peer: its sends to it, its receives from it, run so far
  uint64_t splits;   // its ncclCommSplit calls so far
  ncclResult_t asyncError;
};
typedef struct ncclComm *ncclComm_t;

// The kinds, numbered as `Device.collectiveNs` takes them (a receive is its send's).
enum { K_ALLREDUCE, K_BROADCAST, K_REDUCE, K_ALLGATHER, K_REDUCESCATTER, K_ALLTOALL, K_SEND, K_RECV };
static const char *const kindNames[] = {"AllReduce", "Broadcast", "Reduce", "AllGather", "ReduceScatter",
                                         "AlltoAll", "Send", "Recv"};

struct Task {
  int kind;
  ncclComm_t comm;
  const void *send;
  void *recv;
  size_t count;
  ncclDataType_t type;
  ncclRedOp_t op;
  int root;  // or the peer of a send or receive
  uint64_t seq;   // its number among its comm's like work, given as it runs
  int captured;   // enqueued into a stream capture: a graph's node, run at each replay, kept
};

// The k-th collective of a clique: its ranks' tasks as they arrive, done once performed.
typedef struct Round {
  uint64_t seq;
  int arrived, left;
  Task **tasks;
  int done;
  ncclResult_t result;
  struct Round *next;
} Round;

// A send's staged data, until its receive takes it.
typedef struct Staged {
  int from, to;
  uint64_t seq;
  void *data;
  size_t bytes;
  uint64_t sentNs;  // the sender's clock when it sent
  struct Staged *next;
} Staged;

typedef struct Split {
  uint64_t round;
  int arrived;
  int *color, *key;
  Clique **made;  // per rank: its new clique (NULL for NCCL_SPLIT_NOCOLOR)
  int *newRank;
  struct Split *next;
} Split;

struct Clique {
  ncclUniqueId id;
  int nranks;
  pthread_mutex_t lock;
  pthread_cond_t changed;
  Round *rounds;
  Staged *staged;
  Split *splits;
  Clique *next;
};
static Clique *cliques;
static pthread_mutex_t cliqueLock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t ids;

static Clique *cliqueOf(const ncclUniqueId *id, int nranks) {
  pthread_mutex_lock(&cliqueLock);
  Clique *c = cliques;
  while (c && memcmp(&c->id, id, sizeof *id) != 0) c = c->next;
  if (!c) {
    c = calloc(1, sizeof *c);
    c->id = *id;
    c->nranks = nranks;
    pthread_mutex_init(&c->lock, NULL);
    pthread_cond_init(&c->changed, NULL);
    c->next = cliques;
    cliques = c;
  }
  pthread_mutex_unlock(&cliqueLock);
  return c->nranks == nranks ? c : NULL;
}

static ncclComm_t commNew(Clique *c, int rank) {
  ncclComm_t m = calloc(1, sizeof *m);
  m->clique = c;
  m->rank = rank;
  CUdevice d = 0;
  cuCtxGetDevice(&d);
  m->device = d;
  m->sendSeq = calloc((size_t)c->nranks, sizeof *m->sendSeq);
  m->recvSeq = calloc((size_t)c->nranks, sizeof *m->recvSeq);
  return m;
}

// ---- Performing a collective (the last rank to arrive, the clique's lock held) ----

static void *hostCopy(const void *p, size_t bytes, ncclResult_t *r) {
  void *b = malloc(bytes ? bytes : 1);
  if (!b || (bytes && nvgpuLeanRead(b, (CUdeviceptr)p, bytes) != CUDA_SUCCESS)) *r = ncclUnhandledCudaError;
  return b;
}
static void deviceWrite(void *p, const void *b, size_t bytes, ncclResult_t *r) {
  if (bytes && nvgpuLeanWrite((CUdeviceptr)p, b, bytes) != CUDA_SUCCESS) *r = ncclUnhandledCudaError;
}

// Rank r's input scaled by a PreMulSum's scalar, in place.
static void premul(const RedOp *o, ncclDataType_t t, void *x, size_t n, ncclResult_t *r) {
  double s = o->scalar;
  if (o->deviceScalar) {
    unsigned char v[8];
    if (nvgpuLeanRead(v, (CUdeviceptr)o->deviceScalar, typeBytes(t)) != CUDA_SUCCESS) *r = ncclUnhandledCudaError;
    s = get(t, v, 0);
  }
  for (size_t i = 0; i < n; i++) put(t, x, i, get(t, x, i) * s);
}

// The reduction of every rank's `offset`..+n elements, into a new host buffer.
static void *reduceAll(Round *rd, int n, size_t offset, size_t count, ncclResult_t *r) {
  Task *t0 = rd->tasks[0];
  ncclDataType_t t = t0->type;
  size_t eb = typeBytes(t), bytes = count * eb;
  ncclRedOp_t op = t0->op;
  const RedOp *pre = op >= ncclNumOps && op < ncclNumOps + MAX_OPS ? &ops[op - ncclNumOps] : NULL;
  ncclRedOp_t base = pre ? pre->base : op;
  void *acc = hostCopy((const char *)rd->tasks[0]->send + offset * eb, bytes, r);
  if (pre) premul(pre, t, acc, count, r);
  for (int k = 1; k < n; k++) {
    void *x = hostCopy((const char *)rd->tasks[k]->send + offset * eb, bytes, r);
    if (pre) premul(pre, t, x, count, r);
    combine(t, base, acc, x, count);
    free(x);
  }
  if (base == ncclAvg)
    for (size_t i = 0; i < count; i++) {
      if (isFloat(t)) put(t, acc, i, get(t, acc, i) / n);
      else if (isSigned(t)) putInt(t, acc, i, (uint64_t)(getInt(t, acc, i) / n));
      else putInt(t, acc, i, (uint64_t)getInt(t, acc, i) / (uint64_t)n);
    }
  return acc;
}

// The ranks' clocks: the collective starts when the last rank's device reaches it, takes its time on
// the links, and every rank's device waits until it ends.
static void timeRound(Round *rd, int n) {
  Task *t0 = rd->tasks[0];
  uint64_t bytes = t0->count * typeBytes(t0->type);
  if (t0->kind == K_ALLGATHER || t0->kind == K_REDUCESCATTER || t0->kind == K_ALLTOALL) bytes *= (uint64_t)n;
  uint64_t start = 0;
  for (int k = 0; k < n; k++) {
    uint64_t t = nvgpuLeanDeviceNs(rd->tasks[k]->comm->device);
    if (t > start) start = t;
  }
  uint64_t cost = nvgpuLeanCollectiveNs(t0->comm->device, t0->kind, n, bytes);
  for (int k = 0; k < n; k++) nvgpuLeanDeviceWaitNs(rd->tasks[k]->comm->device, start + cost);
  if (getenv("NVGPU_LEAN_VERBOSE"))
    fprintf(stderr, "nvgpu-lean NCCL: %s of %llu bytes over %d ranks: from %.3f ms, %.3f ms\n",
            kindNames[t0->kind], (unsigned long long)bytes, n, start / 1e6, cost / 1e6);
}

static ncclResult_t perform(Round *rd, int n) {
  ncclResult_t r = ncclSuccess;
  Task *t0 = rd->tasks[0];
  size_t eb = typeBytes(t0->type), count = t0->count;
  for (int k = 1; k < n; k++)
    if (rd->tasks[k]->kind != t0->kind || rd->tasks[k]->type != t0->type || rd->tasks[k]->count != count)
      return ncclInvalidUsage;  // the ranks disagree on the collective
  switch (t0->kind) {
    case K_ALLREDUCE: {
      void *acc = reduceAll(rd, n, 0, count, &r);
      for (int k = 0; k < n; k++) deviceWrite(rd->tasks[k]->recv, acc, count * eb, &r);
      free(acc);
      break;
    }
    case K_REDUCE: {
      void *acc = reduceAll(rd, n, 0, count, &r);
      deviceWrite(rd->tasks[t0->root]->recv, acc, count * eb, &r);
      free(acc);
      break;
    }
    case K_BROADCAST: {
      void *b = hostCopy(rd->tasks[t0->root]->send, count * eb, &r);
      for (int k = 0; k < n; k++) deviceWrite(rd->tasks[k]->recv, b, count * eb, &r);
      free(b);
      break;
    }
    case K_ALLGATHER:
      for (int s = 0; s < n; s++) {
        void *b = hostCopy(rd->tasks[s]->send, count * eb, &r);
        for (int k = 0; k < n; k++) deviceWrite((char *)rd->tasks[k]->recv + (size_t)s * count * eb, b, count * eb, &r);
        free(b);
      }
      break;
    case K_REDUCESCATTER:
      for (int k = 0; k < n; k++) {
        void *acc = reduceAll(rd, n, (size_t)k * count, count, &r);
        deviceWrite(rd->tasks[k]->recv, acc, count * eb, &r);
        free(acc);
      }
      break;
    case K_ALLTOALL:
      for (int s = 0; s < n; s++) {
        void *b = hostCopy(rd->tasks[s]->send, (size_t)n * count * eb, &r);
        for (int k = 0; k < n; k++)
          deviceWrite((char *)rd->tasks[k]->recv + (size_t)s * count * eb, (char *)b + (size_t)k * count * eb, count * eb, &r);
        free(b);
      }
      break;
  }
  return r;
}

// A stream's host function: its rank's part of a collective, or a send or a receive.
static void runTask(void *p) {
  Task *t = p;
  ncclComm_t m = t->comm;
  Clique *c = m->clique;
  int n = c->nranks;
  ncclResult_t r = ncclSuccess;
  size_t bytes = t->count * typeBytes(t->type);
  pthread_mutex_lock(&c->lock);
  t->seq = t->kind == K_SEND ? m->sendSeq[t->root]++ : t->kind == K_RECV ? m->recvSeq[t->root]++ : m->seq++;
  if (t->kind == K_SEND) {
    Staged *s = calloc(1, sizeof *s);
    s->from = m->rank;
    s->to = t->root;
    s->seq = t->seq;
    s->bytes = bytes;
    s->data = hostCopy(t->send, bytes, &r);
    // Sent once its data has crossed: the sender's device waits that long.
    s->sentNs = nvgpuLeanDeviceNs(m->device);
    nvgpuLeanDeviceWaitNs(m->device, s->sentNs + nvgpuLeanCollectiveNs(m->device, K_SEND, c->nranks, bytes));
    s->next = c->staged;
    c->staged = s;
    pthread_cond_broadcast(&c->changed);
  } else if (t->kind == K_RECV) {
    for (;;) {
      Staged **q = &c->staged;
      while (*q && !((*q)->from == t->root && (*q)->to == m->rank && (*q)->seq == t->seq)) q = &(*q)->next;
      if (*q) {
        Staged *s = *q;
        *q = s->next;
        deviceWrite(t->recv, s->data, s->bytes < bytes ? s->bytes : bytes, &r);
        // Received when the data has crossed, after both ends reached the transfer.
        uint64_t now = nvgpuLeanDeviceNs(m->device);
        nvgpuLeanDeviceWaitNs(m->device, (now > s->sentNs ? now : s->sentNs) +
                                         nvgpuLeanCollectiveNs(m->device, K_SEND, c->nranks, s->bytes));
        free(s->data);
        free(s);
        break;
      }
      pthread_cond_wait(&c->changed, &c->lock);
    }
  } else {
    Round *rd = c->rounds;
    while (rd && rd->seq != t->seq) rd = rd->next;
    if (!rd) {
      rd = calloc(1, sizeof *rd);
      rd->seq = t->seq;
      rd->tasks = calloc((size_t)n, sizeof *rd->tasks);
      rd->left = n;
      rd->next = c->rounds;
      c->rounds = rd;
    }
    rd->tasks[m->rank] = t;
    if (++rd->arrived == n) {
      rd->result = perform(rd, n);
      if (rd->result == ncclSuccess) timeRound(rd, n);
      rd->done = 1;
      pthread_cond_broadcast(&c->changed);
    }
    while (!rd->done) pthread_cond_wait(&c->changed, &c->lock);
    r = rd->result;
    if (--rd->left == 0) {
      Round **q = &c->rounds;
      while (*q != rd) q = &(*q)->next;
      *q = rd->next;
      free(rd->tasks);
      free(rd);
    }
  }
  pthread_mutex_unlock(&c->lock);
  if (r != ncclSuccess) {
    m->asyncError = r;
    fprintf(stderr, "nvgpu-lean NCCL: rank %d: a collective failed (%d)\n", m->rank, r);
  }
  if (!t->captured) free(t);
}

// ---- Groups: a thread's calls between ncclGroupStart and ncclGroupEnd, enqueued at the end ----

typedef struct Pending {
  Task *task;
  CUstream stream;
  struct Pending *next;
} Pending;
static __thread int groupDepth;
static __thread Pending *pending, **pendingTail;  // the list's end (NULL: the list is empty)

// The task as stream-ordered work: captured, if its stream is capturing now (a group's work is
// launched at its end, when a capture may have begun), into a graph that runs it at each replay.
static ncclResult_t launch(Task *t, CUstream s) {
  CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
  t->captured = cuStreamIsCapturing(s, &st) == CUDA_SUCCESS && st != CU_STREAM_CAPTURE_STATUS_NONE;
  return cuLaunchHostFunc(s, runTask, t) == CUDA_SUCCESS ? ncclSuccess : ncclUnhandledCudaError;
}

static ncclResult_t enqueue(Task *t, CUstream s) {
  if (groupDepth > 0) {
    if (!pendingTail) pendingTail = &pending;
    Pending *p = calloc(1, sizeof *p);
    p->task = t;
    p->stream = s;
    *pendingTail = p;
    pendingTail = &p->next;
    return ncclSuccess;
  }
  return launch(t, s);
}


static ncclResult_t submit(int kind, const void *send, void *recv, size_t count, ncclDataType_t type,
                           ncclRedOp_t op, int root, ncclComm_t comm, CUstream stream) {
  if (getenv("NVGPU_LEAN_VERBOSE") && comm)
    fprintf(stderr, "nvgpu-lean NCCL: rank %d of %d: %s of %zu elements of type %d\n", comm->rank,
            comm->clique->nranks, kindNames[kind], count, (int)type);
  if (!comm) return ncclInvalidArgument;
  if (!typeBytes(type)) return ncclInvalidArgument;
  if (op >= ncclNumOps && (op >= ncclNumOps + MAX_OPS || !ops[op - ncclNumOps].used)) return ncclInvalidArgument;
  int n = comm->clique->nranks;
  if ((kind == K_BROADCAST || kind == K_REDUCE || kind == K_SEND || kind == K_RECV) && (root < 0 || root >= n))
    return ncclInvalidArgument;
  Task *t = calloc(1, sizeof *t);
  *t = (Task){kind, comm, send, recv, count, type, op, root, 0, 0};
  return enqueue(t, stream);
}

ncclResult_t ncclGroupStart(void) {
  groupDepth++;
  return ncclSuccess;
}

static ncclResult_t resolveSplits(void);

ncclResult_t ncclGroupEnd(void) {
  if (groupDepth == 0) return ncclInvalidUsage;
  if (--groupDepth > 0) return ncclSuccess;
  ncclResult_t r = resolveSplits();
  // Sends first: a receive in the same group may wait on its stream for one of them. Each task's
  // kind is read before any is launched: a launched task may run, and be freed, at once.
  int n = 0;
  for (Pending *p = pending; p; p = p->next) n++;
  int *isSend = calloc((size_t)n + 1, sizeof *isSend);
  int i = 0;
  for (Pending *p = pending; p; p = p->next) isSend[i++] = p->task->kind == K_SEND;
  for (int sends = 1; sends >= 0; sends--) {
    i = 0;
    for (Pending *p = pending; p; p = p->next, i++)
      if (isSend[i] == sends && launch(p->task, p->stream) != ncclSuccess) r = ncclUnhandledCudaError;
  }
  free(isSend);
  while (pending) {
    Pending *p = pending;
    pending = p->next;
    free(p);
  }
  pendingTail = NULL;
  return r;
}

ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *info) {
  (void)info;
  return ncclGroupEnd();
}

// ---- Communicators ----

ncclResult_t ncclGetVersion(int *version) {
  if (!version) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *id) {
  if (!id) return ncclInvalidArgument;
  memset(id, 0, sizeof *id);
  uint64_t k = __atomic_add_fetch(&ids, 1, __ATOMIC_RELAXED);
  snprintf(id->internal, sizeof id->internal, "nvgpu-lean-nccl %d %llu", (int)getpid(), (unsigned long long)k);
  return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId id, int rank) {
  if (!comm || nranks < 1 || rank < 0 || rank >= nranks) return ncclInvalidArgument;
  Clique *c = cliqueOf(&id, nranks);
  if (!c) return ncclInvalidUsage;
  *comm = commNew(c, rank);
  return ncclSuccess;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId id, int rank, ncclConfig_t *config) {
  (void)config;
  return ncclCommInitRank(comm, nranks, id, rank);
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t *comm, int nranks, int rank, int nId, ncclUniqueId *ids_, ncclConfig_t *config) {
  (void)config;
  if (nId < 1 || !ids_) return ncclInvalidArgument;
  return ncclCommInitRank(comm, nranks, ids_[0], rank);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist) {
  if (!comms || ndev < 1) return ncclInvalidArgument;
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  CUcontext was = NULL;
  cuCtxGetCurrent(&was);
  for (int i = 0; i < ndev; i++) {
    CUcontext ctx;
    if (cuDevicePrimaryCtxRetain(&ctx, devlist ? devlist[i] : i) != CUDA_SUCCESS) return ncclUnhandledCudaError;
    cuCtxSetCurrent(ctx);
    ncclResult_t r = ncclCommInitRank(&comms[i], ndev, id, i);
    if (r != ncclSuccess) return r;
  }
  cuCtxSetCurrent(was);
  return ncclSuccess;
}

static void commFree(ncclComm_t comm) {
  free(comm->sendSeq);
  free(comm->recvSeq);
  free(comm);
}
ncclResult_t ncclCommFinalize(ncclComm_t comm) { return comm ? ncclSuccess : ncclInvalidArgument; }
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm) commFree(comm);
  return ncclSuccess;
}
ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

// ncclCommSplit: every rank of the parent calls it; ranks of one color form a clique, ranked by
// key (then by parent rank). Within a group, resolved at its end; outside one, the call waits for
// every rank's.
#define NCCL_SPLIT_NOCOLOR -1
typedef struct SplitCall {
  ncclComm_t parent, *out;
  Split *split;
  struct SplitCall *next;
} SplitCall;
static __thread SplitCall *splitCalls;

static ncclResult_t splitWait(SplitCall *s) {
  Clique *c = s->parent->clique;
  int n = c->nranks, me = s->parent->rank;
  pthread_mutex_lock(&c->lock);
  Split *sp = s->split;
  while (sp->arrived < n) pthread_cond_wait(&c->changed, &c->lock);
  if (!sp->made) {
    // The last to look makes every color's clique.
    sp->made = calloc((size_t)n, sizeof *sp->made);
    sp->newRank = calloc((size_t)n, sizeof *sp->newRank);
    for (int r = 0; r < n; r++) {
      if (sp->color[r] == NCCL_SPLIT_NOCOLOR || sp->made[r]) continue;
      int size = 0;
      for (int q = 0; q < n; q++) size += sp->color[q] == sp->color[r];
      ncclUniqueId id;
      ncclGetUniqueId(&id);
      Clique *nc = cliqueOf(&id, size);
      for (int q = 0; q < n; q++) {
        if (sp->color[q] != sp->color[r]) continue;
        int rank = 0;
        for (int o = 0; o < n; o++)
          if (sp->color[o] == sp->color[q] && (sp->key[o] < sp->key[q] || (sp->key[o] == sp->key[q] && o < q))) rank++;
        sp->made[q] = nc;
        sp->newRank[q] = rank;
      }
    }
  }
  Clique *nc = sp->made[me];
  int rank = sp->newRank[me];
  pthread_mutex_unlock(&c->lock);
  *s->out = nc ? commNew(nc, rank) : NULL;
  if (*s->out) (*s->out)->device = s->parent->device;
  return ncclSuccess;
}

static ncclResult_t resolveSplits(void) {
  ncclResult_t r = ncclSuccess;
  while (splitCalls) {
    SplitCall *s = splitCalls;
    splitCalls = s->next;
    if (splitWait(s) != ncclSuccess) r = ncclInternalError;
    free(s);
  }
  return r;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  (void)config;
  if (!comm || !newcomm) return ncclInvalidArgument;
  Clique *c = comm->clique;
  int n = c->nranks;
  uint64_t round = comm->splits++;
  pthread_mutex_lock(&c->lock);
  Split *sp = c->splits;
  while (sp && sp->round != round) sp = sp->next;
  if (!sp) {
    sp = calloc(1, sizeof *sp);
    sp->round = round;
    sp->color = calloc((size_t)n, sizeof *sp->color);
    sp->key = calloc((size_t)n, sizeof *sp->key);
    sp->next = c->splits;
    c->splits = sp;
  }
  sp->color[comm->rank] = color;
  sp->key[comm->rank] = key;
  sp->arrived++;
  pthread_cond_broadcast(&c->changed);
  pthread_mutex_unlock(&c->lock);
  SplitCall *s = calloc(1, sizeof *s);
  *s = (SplitCall){comm, newcomm, sp, NULL};
  if (groupDepth > 0) {
    s->next = splitCalls;
    splitCalls = s;
    return ncclSuccess;
  }
  ncclResult_t r = splitWait(s);
  free(s);
  return r;
}

ncclResult_t ncclCommShrink(ncclComm_t comm, int *exclude, int nexclude, ncclComm_t *newcomm, ncclConfig_t *config, int flags) {
  (void)flags;
  if (!comm) return ncclInvalidArgument;
  int out = 0;
  for (int i = 0; i < nexclude; i++) out |= exclude[i] == comm->rank;
  return ncclCommSplit(comm, out ? NCCL_SPLIT_NOCOLOR : 0, comm->rank, newcomm, config);
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
  if (!comm || !count) return ncclInvalidArgument;
  *count = comm->clique->nranks;
  return ncclSuccess;
}
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
  if (!comm || !device) return ncclInvalidArgument;
  *device = comm->device;
  return ncclSuccess;
}
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
  if (!comm || !rank) return ncclInvalidArgument;
  *rank = comm->rank;
  return ncclSuccess;
}
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *e) {
  if (!comm || !e) return ncclInvalidArgument;
  *e = comm->asyncError;
  return ncclSuccess;
}

const char *ncclGetErrorString(ncclResult_t r) {
  switch (r) {
    case ncclSuccess: return "no error";
    case ncclUnhandledCudaError: return "unhandled cuda error";
    case ncclSystemError: return "unhandled system error";
    case ncclInternalError: return "internal error";
    case ncclInvalidArgument: return "invalid argument";
    case ncclInvalidUsage: return "invalid usage";
    case ncclRemoteError: return "remote process exited or there was a network error";
    case ncclInProgress: return "NCCL operation in progress";
    default: return "unknown result code";
  }
}
const char *ncclGetLastError(ncclComm_t comm) {
  (void)comm;
  return "";
}

// ---- Reduction operators ----

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t type,
                                      ncclScalarResidence_t residence, ncclComm_t comm) {
  (void)comm;
  if (!op || !scalar || !typeBytes(type)) return ncclInvalidArgument;
  pthread_mutex_lock(&opLock);
  for (int i = 0; i < MAX_OPS; i++)
    if (!ops[i].used) {
      ops[i] = (RedOp){ncclSum, 1, residence == ncclScalarHostImmediate ? get(type, scalar, 0) : 0,
                       residence == ncclScalarHostImmediate ? NULL : scalar, type, 1};
      *op = (ncclRedOp_t)(ncclNumOps + i);
      pthread_mutex_unlock(&opLock);
      return ncclSuccess;
    }
  pthread_mutex_unlock(&opLock);
  return ncclInternalError;
}
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  (void)comm;
  if (op < ncclNumOps || op >= ncclNumOps + MAX_OPS) return ncclInvalidArgument;
  ops[op - ncclNumOps].used = 0;
  return ncclSuccess;
}

// ---- Collectives and point-to-point ----

ncclResult_t ncclAllReduce(const void *s, void *r, size_t n, ncclDataType_t t, ncclRedOp_t op, ncclComm_t c, CUstream st) {
  return submit(K_ALLREDUCE, s, r, n, t, op, 0, c, st);
}
ncclResult_t ncclReduce(const void *s, void *r, size_t n, ncclDataType_t t, ncclRedOp_t op, int root, ncclComm_t c, CUstream st) {
  return submit(K_REDUCE, s, r, n, t, op, root, c, st);
}
ncclResult_t ncclBroadcast(const void *s, void *r, size_t n, ncclDataType_t t, int root, ncclComm_t c, CUstream st) {
  return submit(K_BROADCAST, s, r, n, t, ncclSum, root, c, st);
}
ncclResult_t ncclBcast(void *b, size_t n, ncclDataType_t t, int root, ncclComm_t c, CUstream st) {
  return submit(K_BROADCAST, b, b, n, t, ncclSum, root, c, st);
}
ncclResult_t ncclAllGather(const void *s, void *r, size_t n, ncclDataType_t t, ncclComm_t c, CUstream st) {
  return submit(K_ALLGATHER, s, r, n, t, ncclSum, 0, c, st);
}
ncclResult_t ncclReduceScatter(const void *s, void *r, size_t n, ncclDataType_t t, ncclRedOp_t op, ncclComm_t c, CUstream st) {
  return submit(K_REDUCESCATTER, s, r, n, t, op, 0, c, st);
}
ncclResult_t ncclAlltoAll(const void *s, void *r, size_t n, ncclDataType_t t, ncclComm_t c, CUstream st) {
  return submit(K_ALLTOALL, s, r, n, t, ncclSum, 0, c, st);
}
ncclResult_t ncclSend(const void *s, size_t n, ncclDataType_t t, int peer, ncclComm_t c, CUstream st) {
  return submit(K_SEND, s, NULL, n, t, ncclSum, peer, c, st);
}
ncclResult_t ncclRecv(void *r, size_t n, ncclDataType_t t, int peer, ncclComm_t c, CUstream st) {
  return submit(K_RECV, NULL, r, n, t, ncclSum, peer, c, st);
}

// ---- Memory: device memory as cuMemAlloc gives it; registration is not needed here ----

ncclResult_t ncclMemAlloc(void **p, size_t size) {
  CUdeviceptr d;
  if (cuMemAlloc(&d, size ? size : 1) != CUDA_SUCCESS) return ncclUnhandledCudaError;
  *p = (void *)d;
  return ncclSuccess;
}
ncclResult_t ncclMemFree(void *p) {
  return cuMemFree((CUdeviceptr)p) == CUDA_SUCCESS ? ncclSuccess : ncclUnhandledCudaError;
}
ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size, void **handle) {
  (void)comm; (void)buff; (void)size;
  *handle = NULL;
  return ncclSuccess;
}
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  (void)comm; (void)handle;
  return ncclSuccess;
}
