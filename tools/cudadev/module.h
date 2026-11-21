// Modules, libraries and functions of the emulated driver. Included by cuda.c.
//
// An image is a cubin, a fat binary (or the runtime's wrapper of one, `__fatBinC_Wrapper_t`, as
// cudart's `__cudaRegisterFatBinary` hands it over), or PTX text. The driver's choice is the
// cubin for the device's architecture: an `sm_XYa` cubin only on that architecture, an `sm_XY`
// cubin on any device of major X and minor at least Y; else the newest PTX not newer than the
// device, compiled with `ptxas` for the device's target, as the driver JITs it. The cubin is
// listed with `cuobjdump` (its `-sass` and `-elf` listings), and the Lean side reads both
// once per module (`SASS.Read.Module`).
//
// What an image becomes is kept on disk under the cache directory, by the image's hash and the
// device's target, so a process loading a library's modules (a library may hold thousands) prepares each
// once on the machine. A module loaded with cuModuleLoadData is read at once (its errors are the
// load's); a library (cuLibraryLoadData, how cudart registers every fat binary of a program and of
// the libraries it links) is read when one of its kernels or variables is first used: a program
// uses few of the kernels it registers.

#define WRAPPER_MAGIC 0x466243b1u  // `__fatBinC_Wrapper_t` (crt/fatbinary_section.h)
#define FATBIN_MAGIC 0xBA55ED50u   // the fat binary's header (fatbinary.h)

typedef struct {
  unsigned int magic;
  unsigned int version;
  const void *data;  // the fat binary
  const void *filenameOrFatbins;
} FatbincWrapper;

struct CUfunc_st;

struct CUmod_st {
  char dir[640];            // the image's directory in the cache (named once first used)
  int ready;                // listings in `dir`
  const void *image;        // until ready: the image, its kind ('e', 'f', 'p') and bytes
  int kind;
  size_t size;
  unsigned char *copy;      // the image copied, when the caller does not keep it
  int open[MAX_GPUS];       // read by device g's Lean side, as its module `index[g]`
  uint32_t index[MAX_GPUS];
  CUresult failed;          // a failed preparation, reported at every later use
  struct CUfunc_st *funcs;  // the functions asked for, by name
  // Held while the module is prepared and opened: two threads' first uses of it prepare it once
  // (preparation frees the image the other would read). Taken before `lock`, never after it.
  pthread_mutex_t prep;
};

struct CUfunc_st {
  CUmodule mod;
  char *name;
  int k[MAX_GPUS];  // device g's Lean kernel, once made there (-1 before)
  int known;        // the kernel's numbers (the same on every device), read once it is made
  uint32_t paramBytes, nparams, *poff, *psize;
  int regs, shared, local, maxThreads, sm, cluster[3];
  // Attributes the program sets.
  int maxDynShared, nonPortable, required[3], carveout;
  struct CUfunc_st *next;
};

struct CUlib_st {
  CUmodule mod;
};

static uint64_t fnv(const void *p, size_t n, uint64_t h) {
  const unsigned char *b = p;
  for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
  return h;
}

// The image under a wrapper, and its kind and bytes: 'e' a cubin (its size from the section
// header table's end), 'f' a fat binary (its header's 16 bytes plus the size it records), 'p' PTX.
static const void *imageOf(const void *image, int *kind, size_t *n) {
  const unsigned char *p = image;
  if (*(const uint32_t *)p == WRAPPER_MAGIC) p = ((const FatbincWrapper *)image)->data;
  if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
    const Elf64_Ehdr *h = (const Elf64_Ehdr *)p;
    size_t e = h->e_shoff + (size_t)h->e_shnum * h->e_shentsize;
    size_t ph = h->e_phoff + (size_t)h->e_phnum * h->e_phentsize;
    *kind = 'e';
    *n = e > ph ? e : ph;
  } else if (*(const uint32_t *)p == FATBIN_MAGIC) {
    *kind = 'f';
    *n = 16 + *(const uint64_t *)(p + 8);
  } else {
    *kind = 'p';
    *n = strlen((const char *)p);
  }
  return p;
}

static int run(const char *cmd) {
  if (getenv("NVGPU_LEAN_VERBOSE")) fprintf(stderr, "nvgpu-lean: %s\n", cmd);
  int r = system(cmd);
  return r == 0 ? 0 : -1;
}

static char *slurp(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *b = malloc(n + 1);
  if (fread(b, 1, n, f) != (size_t)n) n = 0;
  b[n] = 0;
  fclose(f);
  if (len) *len = n;
  return b;
}

static int exists(const char *dir, const char *file) {
  char path[800];
  snprintf(path, sizeof path, "%s/%s", dir, file);
  struct stat st;
  return stat(path, &st) == 0;
}

static int mkdirs(const char *path) {
  char p[800];
  snprintf(p, sizeof p, "%s", path);
  for (char *s = p + 1; *s; s++)
    if (*s == '/') {
      *s = 0;
      mkdir(p, 0755);
      *s = '/';
    }
  return mkdir(p, 0755) == 0 || errno == EEXIST ? 0 : -1;
}

// An architecture name's number and suffix: `sm_90a` is (90, 'a'), `compute_80` (80, 0).
static int archOf(const char *s, char *suffix) {
  const char *u = strrchr(s, '_');
  if (!u) return -1;
  // Parsed by hand: glibc 2.38's headers turn atoi into __isoc23_strtol, which older C libraries
  // (a GPU node's) lack.
  int v = 0;
  const char *e = u + 1;
  for (; *e >= '0' && *e <= '9'; e++) v = 10 * v + (*e - '0');
  *suffix = (*e == 'a' || *e == 'f') ? *e : 0;
  return v;
}

// Whether code for architecture (v, suffix) runs on the device: a cubin (`elf`) of the device's
// major and an older or equal minor (`a`: that architecture only; `f`: its family, which is its
// major); PTX of an older or equal architecture (`a`: the same one).
static int fits(int v, char suffix, int elf) {
  char ds;
  int d = archOf(G.part->target, &ds);
  if (suffix == 'a') return v == d && ds == 'a';
  if (elf || suffix == 'f') return v / 10 == d / 10 && v <= d;
  return v <= d;
}

// Among `cuobjdump -lelf` or `-lptx` lines (`ELF file    1: x.1.sm_90a.cubin`), the name of the
// newest fitting one: an exact `a` match first.
static int choose(const char *listing, int elf, char *best, size_t len) {
  int bestV = -1, bestA = 0;
  const char *l = listing;
  while (l && *l) {
    const char *eol = strchr(l, '\n');
    const char *colon = strstr(l, ": ");
    if (colon && (!eol || colon < eol)) {
      char name[512];
      size_t n = (eol ? (size_t)(eol - colon) : strlen(colon)) - 2;
      if (n >= sizeof name) n = sizeof name - 1;
      memcpy(name, colon + 2, n);
      name[n] = 0;
      while (n && (name[n - 1] == ' ' || name[n - 1] == '\r')) name[--n] = 0;
      char *dot = strrchr(name, '.');
      if (dot) {
        *dot = 0;  // the architecture is the last dotted part before the extension
        char suffix;
        int v = archOf(strrchr(name, '.') ? strrchr(name, '.') + 1 : name, &suffix);
        *dot = '.';
        if (v > 0 && fits(v, suffix, elf) && (v > bestV || (v == bestV && suffix == 'a' && !bestA))) {
          bestV = v;
          bestA = suffix == 'a';
          snprintf(best, len, "%s", name);
        }
      }
    }
    l = eol ? eol + 1 : NULL;
  }
  return bestV;
}

// The command's standard output (NULL if it failed).
static char *capture(const char *cmd) {
  if (getenv("NVGPU_LEAN_VERBOSE")) fprintf(stderr, "nvgpu-lean: %s\n", cmd);
  FILE *f = popen(cmd, "r");
  if (!f) return NULL;
  size_t cap = 4096, n = 0;
  char *b = malloc(cap);
  size_t k;
  while ((k = fread(b + n, 1, cap - n - 1, f)) > 0) {
    n += k;
    if (n + 1 == cap) b = realloc(b, cap *= 2);
  }
  b[n] = 0;
  if (pclose(f) != 0) {
    free(b);
    return NULL;
  }
  return b;
}

// Whether the cubin at `path` is in the ELF format before CUDA 13's: OS/ABI 0x33 (the new 0x41).
static int oldElf(const char *path) {
  unsigned char h[8];
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  size_t n = fread(h, 1, sizeof h, f);
  fclose(f);
  return n == sizeof h && memcmp(h, "\x7f" "ELF", 4) == 0 && h[7] == 0x33;
}

// Prepare `w` (a fresh work directory holding `image`, of kind `kind`): the device's cubin as
// m.cubin, and its listings m.sass and m.elf.
static CUresult prepare(const char *w, int kind) {
  char cmd[3000], name[512];
  if (kind == 'e') {
    snprintf(cmd, sizeof cmd, "cd '%s' && mv image m.cubin", w);
    if (run(cmd)) return CUDA_ERROR_INVALID_IMAGE;
  } else if (kind == 'p') {
    snprintf(cmd, sizeof cmd, "cd '%s' && '%s/bin/ptxas' -arch=%s -m64 image -o m.cubin", w, toolkit,
             G.part->target);
    if (run(cmd)) {
      note("ptxas could not compile the module for %s", G.part->target);
      return CUDA_ERROR_INVALID_PTX;
    }
  } else {
    snprintf(cmd, sizeof cmd, "cd '%s' && '%s/bin/cuobjdump' -lelf image 2>/dev/null", w, toolkit);
    char *elfs = capture(cmd);
    if (elfs && choose(elfs, 1, name, sizeof name) > 0) {
      snprintf(cmd, sizeof cmd,
               "cd '%s' && '%s/bin/cuobjdump' -xelf '%s' image >/dev/null && mv '%s' m.cubin", w,
               toolkit, name, name);
      free(elfs);
      if (run(cmd)) return CUDA_ERROR_INVALID_IMAGE;
    } else {
      free(elfs);
      snprintf(cmd, sizeof cmd, "cd '%s' && '%s/bin/cuobjdump' -lptx image 2>/dev/null", w, toolkit);
      char *ptxs = capture(cmd);
      int v = ptxs ? choose(ptxs, 0, name, sizeof name) : -1;
      free(ptxs);
      if (v <= 0) {
        note("the fat binary has no cubin or PTX for %s", G.part->target);
        return CUDA_ERROR_NO_BINARY_FOR_GPU;
      }
      snprintf(cmd, sizeof cmd,
               "cd '%s' && '%s/bin/cuobjdump' -xptx '%s' image >/dev/null && "
               "'%s/bin/ptxas' -arch=%s -m64 '%s' -o m.cubin",
               w, toolkit, name, toolkit, G.part->target, name);
      if (run(cmd)) {
        note("ptxas could not compile %s for %s", name, G.part->target);
        return CUDA_ERROR_INVALID_PTX;
      }
    }
  }
  // A cubin in the ELF format before CUDA 13's (OS/ABI 0x33, ABI version 7, which some CUDA 13
  // libraries still ship) is listed by a CUDA 12 cuobjdump: CUDA 13's finds no device code in it.
  char path[700], dump[700];
  snprintf(path, sizeof path, "%s/m.cubin", w);
  snprintf(dump, sizeof dump, "%s/bin/cuobjdump", toolkit);
  if (oldElf(path)) {
    const char *old = getenv("NVGPU_LEAN_CUOBJDUMP_OLD");
    if (!old || !*old) {
      note("a cubin in the ELF format before CUDA 13's: NVGPU_LEAN_CUOBJDUMP_OLD names a CUDA 12 "
           "cuobjdump to list it (tools/cudadev/cuobjdump.sh)");
      return CUDA_ERROR_INVALID_IMAGE;
    }
    snprintf(dump, sizeof dump, "%s", old);
  }
  snprintf(cmd, sizeof cmd, "cd '%s' && '%s' -sass m.cubin > m.sass && '%s' -elf m.cubin > m.elf", w,
           dump, dump);
  if (run(cmd) == 0) return CUDA_SUCCESS;
  // A cubin newer than the toolkit's cuobjdump reads (nvJitLink's, from a later CUDA release):
  // listed by the newer one NVGPU_LEAN_CUOBJDUMP_NEW names.
  const char *fresh = getenv("NVGPU_LEAN_CUOBJDUMP_NEW");
  if (fresh && *fresh) {
    snprintf(cmd, sizeof cmd, "cd '%s' && '%s' -sass m.cubin > m.sass && '%s' -elf m.cubin > m.elf", w,
             fresh, fresh);
    if (run(cmd) == 0) return CUDA_SUCCESS;
  }
  note("cuobjdump could not list the module's cubin (a newer CUDA release's? NVGPU_LEAN_CUOBJDUMP_NEW "
       "names a newer cuobjdump)");
  return CUDA_ERROR_INVALID_IMAGE;
}

// The module's listings in its cache directory (named by its image's hash and the device's target,
// found there if any process prepared them before), else from its image: prepared in a work
// directory beside it and renamed into place, so processes sharing the cache see a directory
// complete or not at all.
static CUresult prepareModule(CUmodule m) {
  if (m->ready) return CUDA_SUCCESS;
  if (m->failed) return m->failed;
  if (!m->dir[0])
    snprintf(m->dir, sizeof m->dir, "%s/%016llx-%s", cacheDir,
             (unsigned long long)fnv(m->image, m->size, 0xcbf29ce484222325ull ^ m->size), G.part->target);
  if (exists(m->dir, "m.elf")) {
    m->ready = 1;
    free(m->copy);
    m->copy = NULL;
    m->image = NULL;
    return CUDA_SUCCESS;
  }
  char w[700], path[760], cmd[1600];
  snprintf(w, sizeof w, "%s.%d.%lx", m->dir, (int)getpid(), (unsigned long)pthread_self());
  snprintf(path, sizeof path, "%s/image", w);
  FILE *f = mkdirs(w) == 0 ? fopen(path, "wb") : NULL;
  CUresult r = CUDA_SUCCESS;
  if (!f || fwrite(m->image, 1, m->size, f) != m->size) {
    note("cannot write the module cache %s", w);
    r = CUDA_ERROR_OPERATING_SYSTEM;
  }
  if (f) fclose(f);
  if (r == CUDA_SUCCESS) r = prepare(w, m->kind);
  if (r == CUDA_SUCCESS && rename(w, m->dir) != 0 && !exists(m->dir, "m.elf"))
    r = CUDA_ERROR_OPERATING_SYSTEM;
  snprintf(cmd, sizeof cmd, "rm -rf '%s'", w);
  run(cmd);
  if (r != CUDA_SUCCESS) return m->failed = r;
  m->ready = 1;
  free(m->copy);
  m->copy = NULL;
  m->image = NULL;
  return CUDA_SUCCESS;
}

// A module for `image`, which it keeps until it is prepared (`preserved`: the caller keeps it
// alive, as cudart does its fat binaries; else a copy). Nothing else is done until it is used: a
// process registers every fat binary of the libraries it links (hundreds of megabytes for a math
// library or PyTorch) and uses few.
static CUresult newModule(CUmodule *out, const void *image, int preserved) {
  int kind;
  size_t n;
  const void *p = imageOf(image, &kind, &n);
  struct CUmod_st *m = calloc(1, sizeof *m);
  pthread_mutex_init(&m->prep, NULL);
  m->kind = kind;
  m->size = n;
  if (preserved)
    m->image = p;
  else {
    m->copy = malloc(n + 1);
    memcpy(m->copy, p, n);
    m->copy[n] = 0;
    m->image = m->copy;
  }
  *out = m;
  return CUDA_SUCCESS;
}


// The module read by the Lean side (once): its listings, its data placed in the module region.
static CUresult openModuleOnce(CUmodule m);
static CUresult openModule(CUmodule m) {
  pthread_mutex_lock(&m->prep);
  CUresult r = m->open[gcur] ? CUDA_SUCCESS : openModuleOnce(m);
  pthread_mutex_unlock(&m->prep);
  return r;
}
static CUresult openModuleOnce(CUmodule m) {
  CUresult r = prepareModule(m);
  if (r != CUDA_SUCCESS) return r;
  char path[700];
  snprintf(path, sizeof path, "%s/m.sass", m->dir);
  char *sass = slurp(path, NULL);
  snprintf(path, sizeof path, "%s/m.elf", m->dir);
  char *elf = slurp(path, NULL);
  if (!sass || !elf) {
    free(sass);
    free(elf);
    note("module listings missing in %s", m->dir);
    return m->failed = CUDA_ERROR_INVALID_IMAGE;
  }
  pthread_mutex_lock(&lock);
  if (!m->open[gcur]) {
    enterLean();
    lean_object *ls = lean_mk_string(sass), *le = lean_mk_string(elf);
    if (take(nvgpu_module_open(G.lean, ls, le, G.moduleNext))) {
      m->index[gcur] = G.modulesOpened++;
      m->open[gcur] = 1;
      uint64_t end = nvgpu_module_data_end(lend(), m->index[gcur], G.moduleNext);
      G.moduleNext = (end + 255) & ~(uint64_t)255;
    } else
      m->failed = CUDA_ERROR_INVALID_IMAGE;
  }
  pthread_mutex_unlock(&lock);
  free(sass);
  free(elf);
  return m->open[gcur] ? CUDA_SUCCESS : m->failed;
}

static uint64_t kinfo(uint32_t k, uint32_t what) { return nvgpu_kernel_info(lend(), k, what); }

// The function's kernel made (the module opened first), its numbers read.
static CUresult resolve(CUfunction f) {
  if (f->k[gcur] >= 0) return CUDA_SUCCESS;
  CUresult r = openModule(f->mod);
  if (r != CUDA_SUCCESS) return r;
  pthread_mutex_lock(&lock);
  if (f->k[gcur] < 0) {
    enterLean();
    lean_inc(G.lean);
    uint32_t k = nvgpu_kernel_count(G.lean);
    if (!take(nvgpu_module_kernel(G.lean, f->mod->index[gcur], lean_mk_string(f->name))))
      r = CUDA_ERROR_NOT_FOUND;
    else {
      if (!f->known) {
        f->paramBytes = (uint32_t)kinfo(k, 0);
        f->nparams = (uint32_t)kinfo(k, 1);
        f->poff = malloc(sizeof(uint32_t) * (f->nparams + 1));
        f->psize = malloc(sizeof(uint32_t) * (f->nparams + 1));
        for (uint32_t i = 0; i < f->nparams; i++) {
          f->poff[i] = (uint32_t)kinfo(k, 5 + 2 * i);
          f->psize[i] = (uint32_t)kinfo(k, 6 + 2 * i);
        }
        f->shared = (int)kinfo(k, 2);
        f->regs = (int)kinfo(k, 3);
        f->local = (int)kinfo(k, 4);
        f->maxThreads = (int)nvgpu_kernel_attr(lend(), k, 0);
        for (int i = 0; i < 3; i++) f->cluster[i] = (int)nvgpu_kernel_attr(lend(), k, 1 + i);
        f->sm = (int)nvgpu_kernel_attr(lend(), k, 4);
        f->known = 1;
      }
      f->k[gcur] = (int)k;
    }
  }
  pthread_mutex_unlock(&lock);
  return r;
}

// The module's function `name` (one per name), made now with `now`.
static CUresult getFunction(CUfunction *out, CUmodule m, const char *name, int now) {
  if (!m || !name) return CUDA_ERROR_INVALID_VALUE;
  pthread_mutex_lock(&lock);
  CUfunction f = m->funcs;
  while (f && strcmp(f->name, name) != 0) f = f->next;
  if (!f) {
    f = calloc(1, sizeof *f);
    f->mod = m;
    f->name = strdup(name);
    for (int g = 0; g < MAX_GPUS; g++) f->k[g] = -1;
    f->maxDynShared = -1;
    f->carveout = -1;
    f->next = m->funcs;
    m->funcs = f;
  }
  pthread_mutex_unlock(&lock);
  if (now || m->open[gcur]) {
    CUresult r = resolve(f);
    if (r != CUDA_SUCCESS) {
      note("no kernel %s in the module", name);
      return r;
    }
  }
  *out = f;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoadData(CUmodule *m, const void *image) {
  NEED_INIT();
  if (!m || !image) return CUDA_ERROR_INVALID_VALUE;
  CUresult r = newModule(m, image, 1);
  if (r == CUDA_SUCCESS) r = openModule(*m);  // the caller may free the image on return
  return r;
}
CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *m, const void *image, unsigned int n,
                                    CUjit_option *o, void **v) {
  (void)n; (void)o; (void)v;
  return cuModuleLoadData(m, image);
}
CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *m, const void *image) {
  return cuModuleLoadData(m, image);
}
CUresult CUDAAPI cuModuleLoad(CUmodule *m, const char *path) {
  size_t n;
  char *b = slurp(path, &n);
  if (!b) return CUDA_ERROR_FILE_NOT_FOUND;
  CUresult r = cuModuleLoadData(m, b);
  free(b);
  return r;
}
// The module's kernels and data stay on the device (their indices are the Lean side's); the
// handle is the program's to drop.
CUresult CUDAAPI cuModuleUnload(CUmodule m) {
  (void)m;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuModuleGetFunction(CUfunction *f, CUmodule m, const char *name) {
  NEED_INIT();
  return getFunction(f, m, name, 1);
}

static CUresult getGlobal(CUdeviceptr *p, size_t *n, CUmodule m, const char *name) {
  CUresult r = openModule(m);
  if (r != CUDA_SUCCESS) return r;
  pthread_mutex_lock(&lock);
  uint64_t a = nvgpu_module_global(lend(), m->index[gcur], lean_mk_string(name), 0);
  uint64_t size = nvgpu_module_global(lend(), m->index[gcur], lean_mk_string(name), 1);
  pthread_mutex_unlock(&lock);
  if (!a) {
    note("no variable %s in the module", name);
    return CUDA_ERROR_NOT_FOUND;
  }
  if (p) *p = a;
  if (n) *n = size;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *p, size_t *n, CUmodule m, const char *name) {
  NEED_INIT();
  return getGlobal(p, n, m, name);
}
CUresult CUDAAPI cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
  *mode = CU_MODULE_LAZY_LOADING;
  return CUDA_SUCCESS;
}

// ---- Libraries (the context-independent loading API cudart 12 on uses): a library is a lazily
// read module, a kernel its function ----

CUresult CUDAAPI cuLibraryLoadData(CUlibrary *lib, const void *code, CUjit_option *jo, void **jv,
                                   unsigned int nj, CUlibraryOption *lo, void **lv, unsigned int nl) {
  (void)jo; (void)jv; (void)nj;
  NEED_INIT();
  if (!lib || !code) return CUDA_ERROR_INVALID_VALUE;
  // Kept without a copy only when the caller says so with a value: CU_LIBRARY_BINARY_IS_PRESERVED
  // named with no value array promises nothing.
  int preserved = 0;
  for (unsigned i = 0; i < nl && lo && lv; i++)
    if (lo[i] == CU_LIBRARY_BINARY_IS_PRESERVED) preserved = (int)(uintptr_t)lv[i];
  CUmodule m;
  CUresult r = newModule(&m, code, preserved);
  if (r != CUDA_SUCCESS) return r;
  struct CUlib_st *l = malloc(sizeof *l);
  l->mod = m;
  *lib = (CUlibrary)l;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLibraryLoadFromFile(CUlibrary *lib, const char *path, CUjit_option *jo, void **jv,
                                       unsigned int nj, CUlibraryOption *lo, void **lv, unsigned int nl) {
  size_t n;
  char *b = slurp(path, &n);
  if (!b) return CUDA_ERROR_FILE_NOT_FOUND;
  CUresult r = cuLibraryLoadData(lib, b, jo, jv, nj, lo, lv, nl);
  free(b);
  return r;
}
CUresult CUDAAPI cuLibraryUnload(CUlibrary lib) {
  free((struct CUlib_st *)lib);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLibraryGetModule(CUmodule *m, CUlibrary lib) {
  CUmodule mod = ((struct CUlib_st *)lib)->mod;
  CUresult r = openModule(mod);
  if (r == CUDA_SUCCESS) *m = mod;
  return r;
}
CUresult CUDAAPI cuLibraryGetKernel(CUkernel *k, CUlibrary lib, const char *name) {
  NEED_INIT();
  CUfunction f;
  CUresult r = getFunction(&f, ((struct CUlib_st *)lib)->mod, name, 0);
  if (r == CUDA_SUCCESS) *k = (CUkernel)f;
  return r;
}
CUresult CUDAAPI cuLibraryGetGlobal(CUdeviceptr *p, size_t *n, CUlibrary lib, const char *name) {
  NEED_INIT();
  return getGlobal(p, n, ((struct CUlib_st *)lib)->mod, name);
}
CUresult CUDAAPI cuLibraryGetManaged(CUdeviceptr *p, size_t *n, CUlibrary lib, const char *name) {
  (void)p; (void)n; (void)lib;
  note("managed variable %s: managed memory is not modelled", name);
  return CUDA_ERROR_NOT_FOUND;
}
CUresult CUDAAPI cuLibraryGetUnifiedFunction(void **f, CUlibrary lib, const char *name) {
  (void)f; (void)lib; (void)name;
  return CUDA_ERROR_NOT_FOUND;
}
CUresult CUDAAPI cuKernelGetFunction(CUfunction *f, CUkernel k) {
  NEED_INIT();
  CUresult r = resolve((CUfunction)k);
  if (r == CUDA_SUCCESS) *f = (CUfunction)k;
  return r;
}
CUresult CUDAAPI cuKernelGetLibrary(CUlibrary *lib, CUkernel k) {
  (void)lib; (void)k;
  note("cuKernelGetLibrary is not provided by the emulated device");
  return CUDA_ERROR_NOT_SUPPORTED;
}

// ---- Function attributes ----

static int sharedOptin(void) {
  int v;
  partAttribute(G.part, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, &v);
  return v;
}

// The most threads a block of `f` can have: its declared bound, the registers a block can hold
// (64 K per block; a warp's registers are allocated in 256s), and 1024.
static int maxThreadsOf(CUfunction f) {
  int m = 1024;
  if (f->maxThreads > 0 && f->maxThreads < m) m = f->maxThreads;
  if (f->regs > 0) {
    int perWarp = ((f->regs * 32) + 255) / 256 * 256;
    int warps = 65536 / perWarp;
    if (warps * 32 < m) m = warps * 32;
  }
  return m;
}

CUresult CUDAAPI cuFuncGetAttribute(int *v, CUfunction_attribute a, CUfunction f) {
  NEED_INIT();
  CUresult r = resolve(f);
  if (r != CUDA_SUCCESS) return r;
  switch (a) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *v = maxThreadsOf(f); break;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: *v = f->shared; break;
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: *v = 0; break;
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: *v = f->local; break;
    case CU_FUNC_ATTRIBUTE_NUM_REGS: *v = f->regs; break;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION:
    case CU_FUNC_ATTRIBUTE_BINARY_VERSION: *v = f->sm; break;
    case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: *v = 0; break;
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
      *v = f->maxDynShared >= 0 ? f->maxDynShared : 48 * 1024 - f->shared;
      break;
    case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: *v = f->carveout; break;
    case CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET: *v = 0; break;
    case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: *v = f->required[0] ? f->required[0] : f->cluster[0]; break;
    case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: *v = f->required[1] ? f->required[1] : f->cluster[1]; break;
    case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: *v = f->required[2] ? f->required[2] : f->cluster[2]; break;
    case CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: *v = f->nonPortable; break;
    case CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE: *v = 0; break;
    default: *v = 0;
  }
  queryTrace("cuFuncGetAttribute(%s, %d) = %d", f->name, (int)a, *v);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncSetAttribute(CUfunction f, CUfunction_attribute a, int v) {
  NEED_INIT();
  CUresult r = resolve(f);
  if (r != CUDA_SUCCESS) return r;
  queryTrace("cuFuncSetAttribute(%s, %d, %d)", f->name, (int)a, v);
  switch (a) {
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
      if (v < 0 || v + f->shared > sharedOptin()) return CUDA_ERROR_INVALID_VALUE;
      f->maxDynShared = v;
      break;
    case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: f->carveout = v; break;
    case CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED: f->nonPortable = v; break;
    case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: f->required[0] = v; break;
    case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT: f->required[1] = v; break;
    case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH: f->required[2] = v; break;
    default: break;
  }
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction f, CUfunc_cache c) {
  (void)f; (void)c;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction f, CUsharedconfig c) {
  (void)f; (void)c;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncGetModule(CUmodule *m, CUfunction f) {
  *m = f->mod;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncGetName(const char **name, CUfunction f) {
  *name = f->name;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncGetParamInfo(CUfunction f, size_t i, size_t *off, size_t *size) {
  CUresult r = resolve(f);
  if (r != CUDA_SUCCESS) return r;
  if (i >= f->nparams) return CUDA_ERROR_INVALID_VALUE;
  if (off) *off = f->poff[i];
  if (size) *size = f->psize[i];
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncIsLoaded(CUfunctionLoadingState *s, CUfunction f) {
  *s = f->k[gcur] >= 0 ? CU_FUNCTION_LOADING_STATE_LOADED : CU_FUNCTION_LOADING_STATE_UNLOADED;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuFuncLoad(CUfunction f) { return resolve(f); }
CUresult CUDAAPI cuKernelGetAttribute(int *v, CUfunction_attribute a, CUkernel k, CUdevice d) {
  (void)d;
  return cuFuncGetAttribute(v, a, (CUfunction)k);
}
CUresult CUDAAPI cuKernelSetAttribute(CUfunction_attribute a, int v, CUkernel k, CUdevice d) {
  (void)d;
  return cuFuncSetAttribute((CUfunction)k, a, v);
}
CUresult CUDAAPI cuKernelSetCacheConfig(CUkernel k, CUfunc_cache c, CUdevice d) {
  (void)k; (void)c; (void)d;
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuKernelGetName(const char **name, CUkernel k) { return cuFuncGetName(name, (CUfunction)k); }
CUresult CUDAAPI cuKernelGetParamInfo(CUkernel k, size_t i, size_t *off, size_t *size) {
  return cuFuncGetParamInfo((CUfunction)k, i, off, size);
}

// ---- Occupancy, from the part's limits: warps, registers (a warp's in 256s), shared memory (each
// block also takes the reserved 1 KB) and blocks per SM ----

static int attr(CUdevice_attribute a) {
  int v;
  partAttribute(G.part, a, &v);
  return v;
}

static int blocksPerSm(CUfunction f, int block, size_t dyn) {
  int warps = (block + 31) / 32;
  if (warps == 0) return 0;
  int n = G.part->maxWarpsPerSm / warps;
  if (G.part->maxBlocksPerSm < n) n = G.part->maxBlocksPerSm;
  if (f->regs > 0) {
    int perWarp = ((f->regs * 32) + 255) / 256 * 256;
    int byRegs = (G.part->regsPerSm / perWarp) / warps;
    if (byRegs < n) n = byRegs;
  }
  size_t smem = f->shared + dyn + attr(CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK);
  if (smem > 0) {
    int bySmem = (int)(attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR) / smem);
    if (bySmem < n) n = bySmem;
  }
  return n;
}

CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *n, CUfunction f, int block,
                                                             size_t smem) {
  NEED_INIT();
  CUresult r = resolve(f);
  if (r != CUDA_SUCCESS) return r;
  *n = block > maxThreadsOf(f) ? 0 : blocksPerSm(f, block, smem);
  queryTrace("cuOccupancyMaxActiveBlocks(%s, %d, %zu) = %d", f->name, block, smem, *n);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *n, CUfunction f,
                                                                      int block, size_t smem,
                                                                      unsigned flags) {
  (void)flags;
  return cuOccupancyMaxActiveBlocksPerMultiprocessor(n, f, block, smem);
}
CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGrid, int *blockOut, CUfunction f,
                                                           CUoccupancyB2DSize smemOf, size_t smem,
                                                           int limit, unsigned int flags) {
  (void)flags;
  NEED_INIT();
  CUresult r = resolve(f);
  if (r != CUDA_SUCCESS) return r;
  int top = maxThreadsOf(f);
  if (limit > 0 && limit < top) top = limit;
  int best = 0, bestBlock = 32;
  for (int b = top / 32 * 32; b >= 32; b -= 32) {
    int n = blocksPerSm(f, b, smemOf ? smemOf(b) : smem);
    if (n * b > best) {
      best = n * b;
      bestBlock = b;
    }
  }
  *blockOut = bestBlock;
  *minGrid = (best / bestBlock) * attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  return CUDA_SUCCESS;
}
CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(int *minGrid, int *block, CUfunction f,
                                                  CUoccupancyB2DSize smemOf, size_t smem, int limit) {
  return cuOccupancyMaxPotentialBlockSizeWithFlags(minGrid, block, f, smemOf, smem, limit, 0);
}
CUresult CUDAAPI cuOccupancyAvailableDynamicSMemPerBlock(size_t *out, CUfunction f, int blocks, int block) {
  NEED_INIT();
  (void)block;
  CUresult r = resolve(f);
  if (r != CUDA_SUCCESS) return r;
  if (blocks <= 0) return CUDA_ERROR_INVALID_VALUE;
  long per = attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR) / blocks -
             attr(CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK) - f->shared;
  long cap = sharedOptin() - f->shared;
  *out = per < 0 ? 0 : (size_t)(per < cap ? per : cap);
  queryTrace("cuOccupancyAvailableDynamicSMem(%s, %d, %d) = %zu", f->name, blocks, block, *out);
  return CUDA_SUCCESS;
}

static void clusterOf(const CUlaunchConfig *c, CUfunction f, unsigned d[3]) {
  d[0] = f->cluster[0] ? f->cluster[0] : 1;
  d[1] = f->cluster[1] ? f->cluster[1] : 1;
  d[2] = f->cluster[2] ? f->cluster[2] : 1;
  for (unsigned i = 0; c && i < c->numAttrs; i++)
    if (c->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
      d[0] = c->attrs[i].value.clusterDim.x;
      d[1] = c->attrs[i].value.clusterDim.y;
      d[2] = c->attrs[i].value.clusterDim.z;
    }
}

// Clusters of the launch's shape that fit on the device at once: a cluster's blocks share a GPC,
// so every GPC holds as many as its SMs' blocks allow (the part's SMs taken as evenly spread over
// its GPCs, the documented partition sizes being unpublished per part).
CUresult CUDAAPI cuOccupancyMaxActiveClusters(int *n, CUfunction f, const CUlaunchConfig *c) {
  NEED_INIT();
  CUresult r = resolve(f);
  if (r != CUDA_SUCCESS) return r;
  unsigned d[3];
  clusterOf(c, f, d);
  int size = (int)(d[0] * d[1] * d[2]);
  int perSm = blocksPerSm(f, (int)(c->blockDimX * c->blockDimY * c->blockDimZ), c->sharedMemBytes);
  int sms = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  *n = size > 0 ? perSm * sms / size : 0;
  queryTrace("cuOccupancyMaxActiveClusters(%s, %u, %u, %u, %zu) = %d", f->name,
             c->blockDimX, c->blockDimY, c->blockDimZ, (size_t)c->sharedMemBytes, *n);
  return CUDA_SUCCESS;
}
// The largest cluster the launch may use: 8 blocks (portable), 16 where the kernel allows a
// non-portable size on a part with clusters (sm_90 on).
CUresult CUDAAPI cuOccupancyMaxPotentialClusterSize(int *n, CUfunction f, const CUlaunchConfig *c) {
  NEED_INIT();
  (void)c;
  CUresult r = resolve(f);
  if (r != CUDA_SUCCESS) return r;
  *n = G.part->ccMajor < 9 ? 1 : f->nonPortable ? 16 : 8;
  return CUDA_SUCCESS;
}
