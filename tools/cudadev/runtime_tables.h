// The driver's tables the CUDA runtime (cudart, and the libraries that link it statically) asks
// for by identifier (`cuGetExportTable`, cuda.h) before anything else: their identifiers, layouts
// and the integrity check, as ZLUDA defines and implements them (github.com/vosen/ZLUDA, commit
// eb89ae3b23b2abaf8480e496fdd5d51f0651d330: dark_api/src/lib.rs, zluda/src/impl/driver.rs;
// its authors' MIT or Apache-2.0 license, this port under MIT: LICENSE-ZLUDA). The
// tables' names are ZLUDA's. Entries ZLUDA does not define fail with CUDA_ERROR_NOT_SUPPORTED.
// Included by cuda.c.

// CUDART_INTERFACE: modules from the runtime's fat-binary wrapper (module.h reads the wrapper),
// and the primary context.
static CUresult cudartModuleFromCubin(CUmodule *m, const FatbincWrapper *w) {
  if (!m || !w) return CUDA_ERROR_INVALID_VALUE;
  return cuModuleLoadData(m, w);
}
static CUresult cudartModuleFromCubinExt1(CUmodule *m, const FatbincWrapper *w, void *a3, void *a4,
                                          unsigned int a5) {
  if (a3 || a4 || a5) return CUDA_ERROR_NOT_SUPPORTED;
  return cudartModuleFromCubin(m, w);
}
static CUresult cudartModuleFromCubinExt2(const void *fatbin, CUmodule *m, void *a3, void *a4,
                                          unsigned int a5) {
  if (a3 || a4 || a5) return CUDA_ERROR_NOT_SUPPORTED;
  if (!m || !fatbin) return CUDA_ERROR_INVALID_VALUE;
  return cuModuleLoadData(m, fatbin);
}
static CUresult cudartPrimaryContext(CUcontext *c, CUdevice d) {
  if (!c) return CUDA_ERROR_INVALID_VALUE;
  return cuDevicePrimaryCtxRetain(c, d);
}
static CUresult cudartPrimaryRetainWithFlags(CUdevice d, unsigned int flags) {
  CUcontext c;
  CUresult r = cuDevicePrimaryCtxRetain(&c, d);
  return r != CUDA_SUCCESS ? r : cuDevicePrimaryCtxSetFlags(d, flags);
}
static void cudartFn7(size_t a) { (void)a; }
static CUresult cudartLoadCompilers(void) { return CUDA_SUCCESS; }

// TOOLS_RUNTIME_CALLBACK_HOOKS: two buffers of 1024 and 14 words.
static uint32_t hookBuffer1[1024], hookBuffer2[14];
static void hooksBuffer1(void **p, size_t *n) { *p = hookBuffer1; *n = 1024; }
static void hooksBuffer2(void **p, size_t *n) { *p = hookBuffer2; *n = 14; }

// CONTEXT_LOCAL_STORAGE_INTERFACE_V0301: values by key; the device has one context.
typedef struct Cls {
  void *key, *value;
  struct Cls *next;
} Cls;
static Cls *cls;
static CUresult clsPut(CUcontext c, void *key, void *value, void *dtor) {
  (void)c; (void)dtor;
  pthread_mutex_lock(&lock);
  Cls *x = cls;
  while (x && x->key != key) x = x->next;
  if (!x) {
    x = malloc(sizeof *x);
    x->key = key;
    x->next = cls;
    cls = x;
  }
  x->value = value;
  pthread_mutex_unlock(&lock);
  return CUDA_SUCCESS;
}
static CUresult clsDelete(CUcontext c, void *key) {
  (void)c;
  pthread_mutex_lock(&lock);
  for (Cls **x = &cls; *x; x = &(*x)->next)
    if ((*x)->key == key) {
      Cls *d = *x;
      *x = d->next;
      free(d);
      break;
    }
  pthread_mutex_unlock(&lock);
  return CUDA_SUCCESS;
}
static CUresult clsGet(void **value, CUcontext c, void *key) {
  (void)c;
  CUresult r = CUDA_ERROR_INVALID_HANDLE;
  pthread_mutex_lock(&lock);
  for (Cls *x = cls; x; x = x->next)
    if (x->key == key) {
      *value = x->value;
      r = CUDA_SUCCESS;
      break;
    }
  pthread_mutex_unlock(&lock);
  return r;
}

// CTX_CREATE_BYPASS: the device's one context.
static CUresult ctxCreateBypass(CUcontext *c, unsigned int flags, CUdevice d) {
  (void)flags;
  if (!c) return CUDA_ERROR_INVALID_VALUE;
  return cuDevicePrimaryCtxRetain(c, d);
}

// HEAP_ACCESS: a record of two words.
typedef struct {
  size_t x, y;
} HeapRecord;
static CUresult heapAlloc(const void **rec, size_t a, size_t b) {
  HeapRecord *r = malloc(sizeof *r);
  r->x = a;
  r->y = b;
  *rec = r;
  return CUDA_SUCCESS;
}
static CUresult heapFree(const void *rec, size_t *b) {
  HeapRecord *r = (HeapRecord *)rec;
  if (b) *b = r->y;
  free(r);
  return CUDA_SUCCESS;
}

// DEVICE_EXTENDED_RT: not provided, as in ZLUDA.
static CUresult deviceAttributeExt(CUdevice d, unsigned int a, int u, size_t *r) {
  (void)d; (void)a; (void)u; (void)r;
  return CUDA_ERROR_NOT_SUPPORTED;
}
static CUresult deviceSomething(unsigned char *r, CUdevice d) {
  (void)r; (void)d;
  return CUDA_ERROR_NOT_SUPPORTED;
}

// CONTEXT_CHECKS.
static CUresult contextCheck(CUcontext c, uint32_t *r1, const void **r2) {
  (void)c; (void)r2;
  *r1 = 0;
  return CUDA_SUCCESS;
}
static uint32_t checkFn3(void) { return 0; }

// INTEGRITY_CHECK: a keyed hash of the tables' addresses, the process, thread, time and devices,
// which the runtime compares with its own computation (ZLUDA's `integrity_check`).
static const uint8_t mixing[256] = {
  0x29, 0x2E, 0x43, 0xC9, 0xA2, 0xD8, 0x7C, 0x01, 0x3D, 0x36, 0x54, 0xA1, 0xEC, 0xF0, 0x06, 0x13,
  0x62, 0xA7, 0x05, 0xF3, 0xC0, 0xC7, 0x73, 0x8C, 0x98, 0x93, 0x2B, 0xD9, 0xBC, 0x4C, 0x82, 0xCA,
  0x1E, 0x9B, 0x57, 0x3C, 0xFD, 0xD4, 0xE0, 0x16, 0x67, 0x42, 0x6F, 0x18, 0x8A, 0x17, 0xE5, 0x12,
  0xBE, 0x4E, 0xC4, 0xD6, 0xDA, 0x9E, 0xDE, 0x49, 0xA0, 0xFB, 0xF5, 0x8E, 0xBB, 0x2F, 0xEE, 0x7A,
  0xA9, 0x68, 0x79, 0x91, 0x15, 0xB2, 0x07, 0x3F, 0x94, 0xC2, 0x10, 0x89, 0x0B, 0x22, 0x5F, 0x21,
  0x80, 0x7F, 0x5D, 0x9A, 0x5A, 0x90, 0x32, 0x27, 0x35, 0x3E, 0xCC, 0xE7, 0xBF, 0xF7, 0x97, 0x03,
  0xFF, 0x19, 0x30, 0xB3, 0x48, 0xA5, 0xB5, 0xD1, 0xD7, 0x5E, 0x92, 0x2A, 0xAC, 0x56, 0xAA, 0xC6,
  0x4F, 0xB8, 0x38, 0xD2, 0x96, 0xA4, 0x7D, 0xB6, 0x76, 0xFC, 0x6B, 0xE2, 0x9C, 0x74, 0x04, 0xF1,
  0x45, 0x9D, 0x70, 0x59, 0x64, 0x71, 0x87, 0x20, 0x86, 0x5B, 0xCF, 0x65, 0xE6, 0x2D, 0xA8, 0x02,
  0x1B, 0x60, 0x25, 0xAD, 0xAE, 0xB0, 0xB9, 0xF6, 0x1C, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7E, 0x0F,
  0x55, 0x47, 0xA3, 0x23, 0xDD, 0x51, 0xAF, 0x3A, 0xC3, 0x5C, 0xF9, 0xCE, 0xBA, 0xC5, 0xEA, 0x26,
  0x2C, 0x53, 0x0D, 0x6E, 0x85, 0x28, 0x84, 0x09, 0xD3, 0xDF, 0xCD, 0xF4, 0x41, 0x81, 0x4D, 0x52,
  0x6A, 0xDC, 0x37, 0xC8, 0x6C, 0xC1, 0xAB, 0xFA, 0x24, 0xE1, 0x7B, 0x08, 0x0C, 0xBD, 0xB1, 0x4A,
  0x78, 0x88, 0x95, 0x8B, 0xE3, 0x63, 0xE8, 0x6D, 0xE9, 0xCB, 0xD5, 0xFE, 0x3B, 0x00, 0x1D, 0x39,
  0xF2, 0xEF, 0xB7, 0x0E, 0x66, 0x58, 0xD0, 0xE4, 0xA6, 0x77, 0x72, 0xF8, 0xEB, 0x75, 0x4B, 0x0A,
  0x31, 0x44, 0x50, 0xB4, 0x8F, 0xED, 0x1F, 0x1A, 0xDB, 0x99, 0x8D, 0x33, 0x9F, 0x11, 0x83, 0x14};

// `integrity_check_single_pass`.
static void mixOne(uint8_t *a, uint8_t b) {
  uint8_t t1 = a[0x40];
  a[t1 + 0x10] = b;
  uint8_t t3 = (uint8_t)((t1 + 1) & 0xf);
  a[t1 + 0x20] = a[t1] ^ b;
  uint8_t t4 = mixing[(uint8_t)(b ^ a[0x41])];
  uint8_t old = a[t1 + 0x30];
  a[t1 + 0x30] = t4 ^ old;
  a[0x41] = t4 ^ old;
  a[0x40] = t3;
  if (t3 != 0) return;
  uint8_t t = 0x29, t5 = 0;
  for (;;) {
    t = t ^ a[0];
    a[0] = t;
    for (int k = 1; k < 0x30; k++) {
      t = a[k] ^ mixing[t];
      a[k] = t;
    }
    t = (uint8_t)(t + t5);
    t5++;
    if (t5 == 0x12) break;
    t = mixing[t];
  }
}

// `hash_pass`.
static void mixBytes(uint8_t *a, const void *p, size_t n, uint8_t mask) {
  for (size_t i = 0; i < n; i++) mixOne(a, ((const uint8_t *)p)[i] ^ mask);
}

// `pass5`.
static void mixFinish(uint8_t *a, uint64_t out[2]) {
  uint8_t t = (uint8_t)(16 - a[64]);
  for (int i = 0; i < t; i++) mixOne(a, t);
  for (int k = 0x30; k < 0x40; k++) mixOne(a, a[k]);
  memcpy(out, a, 16);
}

static void *runtimeTables[10][26];
#define TABLE_CUDART 0
#define TABLE_INTEGRITY 7

static CUresult integrityCheck(uint32_t version, uint64_t unixSeconds, uint64_t result[2]) {
  if (version % 10 == 0) {
    result[0] = 0x3341181C03CB675Cull;
    result[1] = 0x8ED383AA1F4CD1E8ull;
    return CUDA_SUCCESS;
  }
  if (version % 10 == 1) {
    result[0] = 0x1841181C03CB675Cull;
    result[1] = 0x8ED383AA1F4CD1E8ull;
    return CUDA_SUCCESS;
  }
  static const uint8_t pass1[16] = {0x14, 0x6A, 0xDD, 0xAE, 0x53, 0xA9, 0xA7, 0x52,
                                    0xAA, 0x08, 0x41, 0x36, 0x0B, 0xF5, 0x5A, 0x9F};
  uint8_t a[66];
  memset(a, 0, sizeof a);
  mixBytes(a, pass1, 16, 0x36);
  struct {
    uint32_t driverVersion, version, process, thread;
    const void *cudartTable, *integrityTable, *fn;
    uint64_t unixSeconds;
  } in = {CUDA_VERSION, version, (uint32_t)getpid(), (uint32_t)pthread_self(),
          runtimeTables[TABLE_CUDART], runtimeTables[TABLE_INTEGRITY], runtimeTables[TABLE_INTEGRITY][1],
          unixSeconds};
  mixBytes(a, &in, sizeof in, 0);
  // Every device's identity, in ordinal order (`DeviceHashinfo`).
  for (int d = 0; d < ngpus; d++) {
    struct {
      CUuuid guid;
      int32_t domain, bus, device;
    } pci;
    memset(&pci, 0, sizeof pci);
    cuDeviceGetUuid(&pci.guid, d);
    cuDeviceGetAttribute(&pci.domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, d);
    cuDeviceGetAttribute(&pci.bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, d);
    cuDeviceGetAttribute(&pci.device, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, d);
    mixBytes(a, &pci, sizeof pci, 0);
  }
  uint64_t first[2];
  mixFinish(a, first);
  memset(a, 0, 16);
  memset(a + 48, 0, 18);
  mixBytes(a, pass1, 16, 0x5c);
  mixBytes(a, first, 16, 0);
  mixFinish(a, result);
  return CUDA_SUCCESS;
}

// The tables: their names, identifiers (as bytes, in the order the GUID strings print) and
// lengths; entry 0 is the table's size in bytes where marked.
static const struct {
  const char *name;
  uint8_t guid[16];
  int len;
  int sized;
} runtimeIds[] = {
  {"CUDART_INTERFACE", {0x6B, 0xD5, 0xFB, 0x6C, 0x5B, 0xF4, 0xE7, 0x4A, 0x89, 0x87, 0xD9, 0x39, 0x12, 0xFD, 0x9D, 0xF9}, 13, 1},
  {"TOOLS_TLS", {0x42, 0xD8, 0x5A, 0x81, 0x23, 0xF6, 0xCB, 0x47, 0x82, 0x98, 0xF6, 0xE7, 0x8A, 0x3A, 0xEC, 0xDC}, 4, 1},
  {"TOOLS_RUNTIME_CALLBACK_HOOKS", {0xA0, 0x94, 0x79, 0x8C, 0x2E, 0x74, 0x2E, 0x74, 0x93, 0xF2, 0x08, 0x00, 0x20, 0x0C, 0x0A, 0x66}, 7, 1},
  {"CONTEXT_LOCAL_STORAGE_INTERFACE_V0301", {0xC6, 0x93, 0x33, 0x6E, 0x11, 0x21, 0xDF, 0x11, 0xA8, 0xC3, 0x68, 0xF3, 0x55, 0xD8, 0x95, 0x93}, 4, 0},
  {"CTX_CREATE_BYPASS", {0x0C, 0xA5, 0x0B, 0x8C, 0x10, 0x04, 0x92, 0x9A, 0x89, 0xA7, 0xD0, 0xDF, 0x10, 0xE7, 0x72, 0x86}, 2, 1},
  {"HEAP_ACCESS", {0x19, 0x5B, 0xCB, 0xF4, 0xD6, 0x7D, 0x02, 0x4A, 0xAC, 0xC5, 0x1D, 0x29, 0xCE, 0xA6, 0x31, 0xAE}, 3, 1},
  {"DEVICE_EXTENDED_RT", {0xB1, 0x05, 0x41, 0xE1, 0xF7, 0xC7, 0xC7, 0x4A, 0x9F, 0x64, 0xF2, 0x23, 0xBE, 0x99, 0xF1, 0xE2}, 26, 1},
  {"INTEGRITY_CHECK", {0xD4, 0x08, 0x20, 0x55, 0xBD, 0xE6, 0x70, 0x4B, 0x8D, 0x34, 0xBA, 0x12, 0x3C, 0x66, 0xE1, 0xF2}, 3, 1},
  {"CONTEXT_CHECKS", {0x26, 0x3E, 0x88, 0x60, 0x7C, 0xD2, 0x61, 0x43, 0x92, 0xF6, 0xBB, 0xD5, 0x00, 0x6D, 0xFA, 0x7E}, 4, 1},
};
#define RUNTIME_TABLES (sizeof runtimeIds / sizeof runtimeIds[0])

static CUresult runtimeNotSupported(void) { return CUDA_ERROR_NOT_SUPPORTED; }

// The tables filled once, published to every caller only when complete; traced
// (NVGPU_LEAN_TRACE), every entry is handed out through a trampoline named by its table and index
// (the integrity check's hash covers the addresses as handed out, so it stays consistent).
static void runtimeFill(void) {
  for (unsigned t = 0; t < RUNTIME_TABLES; t++)
    if (runtimeIds[t].sized) runtimeTables[t][0] = (void *)(uintptr_t)(runtimeIds[t].len * sizeof(void *));
  void **c = runtimeTables[0];
  c[1] = (void *)cudartModuleFromCubin;
  c[2] = (void *)cudartPrimaryContext;
  c[4] = (void *)cudartPrimaryRetainWithFlags;
  c[6] = (void *)cudartModuleFromCubinExt1;
  c[7] = (void *)cudartFn7;
  c[8] = (void *)cudartModuleFromCubinExt2;
  c[12] = (void *)cudartLoadCompilers;
  runtimeTables[2][2] = (void *)hooksBuffer1;
  runtimeTables[2][6] = (void *)hooksBuffer2;
  runtimeTables[3][0] = (void *)clsPut;
  runtimeTables[3][1] = (void *)clsDelete;
  runtimeTables[3][2] = (void *)clsGet;
  runtimeTables[4][1] = (void *)ctxCreateBypass;
  runtimeTables[5][1] = (void *)heapAlloc;
  runtimeTables[5][2] = (void *)heapFree;
  runtimeTables[6][5] = (void *)deviceAttributeExt;
  runtimeTables[6][13] = (void *)deviceSomething;
  runtimeTables[TABLE_INTEGRITY][1] = (void *)integrityCheck;
  runtimeTables[8][2] = (void *)contextCheck;
  runtimeTables[8][3] = (void *)checkFn3;
  for (unsigned t = 0; t < RUNTIME_TABLES; t++)
    for (int i = runtimeIds[t].sized ? 1 : 0; i < runtimeIds[t].len; i++) {
      int returnsVoid = (t == 0 && i == 7) || (t == 2 && (i == 2 || i == 6));
      if (!runtimeTables[t][i]) { runtimeTables[t][i] = (void *)runtimeNotSupported; returnsVoid = 0; }
      char name[64];
      snprintf(name, sizeof name, "%s[%d]", runtimeIds[t].name, i);
      runtimeTables[t][i] = traced(name, runtimeTables[t][i], returnsVoid);
    }
}
static void runtimeInit(void) {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, runtimeFill);
}

CUresult CUDAAPI cuGetExportTable(const void **table, const CUuuid *id) {
  if (!table || !id) return CUDA_ERROR_INVALID_VALUE;
  runtimeInit();
  for (unsigned t = 0; t < RUNTIME_TABLES; t++)
    if (memcmp(runtimeIds[t].guid, id->bytes, 16) == 0) {
      if (tracing > 1) fprintf(stderr, "nvgpu-lean: cuGetExportTable(%s)\n", runtimeIds[t].name);
      *table = runtimeTables[t];
      return CUDA_SUCCESS;
    }
  if (tracing > 0) {
    char hex[40];
    for (int i = 0; i < 16; i++) snprintf(hex + 2 * i, 3, "%02X", (uint8_t)id->bytes[i]);
    fprintf(stderr, "nvgpu-lean: cuGetExportTable(%s): not provided\n", hex);
  }
  *table = NULL;
  return CUDA_ERROR_INVALID_VALUE;
}
