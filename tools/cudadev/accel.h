// The accelerated backend (cuda.c's part; included there).
//
// With NVGPU_LEAN_ACCEL naming the host GPU's target (sm_80, sm_90a, …), a launch runs on the
// host's own GPU as the accelerated mode's lifted CUDA (tools/lift) instead of in the simulator:
// the device exports it (nvgpu_launch_export: the harness's const0.bin, kernel.jsonl and meta.txt)
// with the device memory's two segments (global.bin: the allocations, from DEVICE_BASE to the end
// of the last; module.bin: the modules' data), the lifter translates the kernel and the host's nvcc
// builds it into the harness (tools/lift/main.cu) once per kernel, block size and lifter, the
// harness runs it, and the device takes back the pages it changed. Functional: it counts no
// cycles. A fault the lifted kernel detects, or a kernel the lifter cannot translate, fails the
// launch.
//
// Environment: NVGPU_LEAN_LIFT (the tools/lift directory; its mufu_tables.cuh is used when there, as
// in a copy without the repository's NV/Num/MufuTables.lean, else generated), NVGPU_LEAN_ACCEL_NVCC (default nvcc)
// and NVGPU_LEAN_ACCEL_NVCC_FLAGS (its further flags: a wheel's include and library directories),
// NVGPU_LEAN_ACCEL_LIBRARY_PATH (the library path the harness runs with, where the host's libcuda
// is; this library is never on it), NVGPU_LEAN_ACCEL_KEEP (keep every launch's directory under the
// cache's accel/, not only a failed one's).

#include <sys/stat.h>
#include <sys/wait.h>

extern lean_object *nvgpu_launch_export(lean_obj_arg d, uint32_t k, uint32_t gx, uint32_t gy,
                                        uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz,
                                        uint32_t cx, uint32_t cy, uint32_t cz, uint32_t smem,
                                        lean_obj_arg params, lean_obj_arg dir, lean_obj_arg extra,
                                        lean_object *w);

#define ACCEL_PAGE 4096

static const char *accelArch(void) { return getenv("NVGPU_LEAN_ACCEL"); }

// FNV-1a over a file's bytes, from `h` (module.h's `fnv` and `slurp`).
static uint64_t fnvFile(const char *path, uint64_t h) {
  size_t n;
  void *p = slurp(path, &n);
  if (!p) return h;
  h = fnv(p, n, h);
  free(p);
  return h;
}

static int spill(const char *path, const void *p, size_t n) {
  FILE *f = fopen(path, "wb");
  if (!f) return 0;
  int ok = fwrite(p, 1, n, f) == n;
  return fclose(f) == 0 && ok;
}

// Run `cmd` through the shell, its output to `log`; on failure the log's end noted.
static int accelRun(const char *what, const char *cmd, const char *log) {
  char *full = malloc(strlen(cmd) + strlen(log) + 32);
  sprintf(full, "%s > '%s' 2>&1", cmd, log);
  int st = system(full);
  free(full);
  if (st != -1 && WIFEXITED(st) && WEXITSTATUS(st) == 0) return 1;
  size_t n = 0;
  char *out = slurp(log, &n);
  size_t from = n > 600 ? n - 600 : 0;
  note("%s failed: %.*s", what, (int)(n - from), out ? out + from : "");
  free(out);
  return 0;
}

// The device's bytes [addr, addr + n) as a Lean byte array (the lock held).
static lean_object *accelRead(CUdeviceptr addr, size_t n) {
  if (n == 0) return lean_alloc_sarray(1, 0, 0);
  return ioValue(nvgpu_read(lend(), addr, n, lean_io_mk_world()));
}

// The pages of `after` that differ from `before`, written to the device at `base`.
static int accelTakeBack(CUdeviceptr base, const uint8_t *before, const uint8_t *after, size_t n) {
  size_t i = 0;
  while (i < n) {
    size_t len = n - i < ACCEL_PAGE ? n - i : ACCEL_PAGE;
    if (!memcmp(before + i, after + i, len)) { i += len; continue; }
    size_t j = i + len;
    while (j < n) {
      size_t l2 = n - j < ACCEL_PAGE ? n - j : ACCEL_PAGE;
      if (!memcmp(before + j, after + j, l2)) break;
      j += l2;
    }
    if (devWrite(base + i, after + i, j - i) != CUDA_SUCCESS) return 0;
    i = j;
  }
  return 1;
}

static void accelRemove(const char *dir) {
  if (getenv("NVGPU_LEAN_ACCEL_KEEP")) return;
  char cmd[700];
  snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
  if (system(cmd) != 0) note("could not remove %s", dir);
}

// The harness built for this launch's kernel (`dir`'s kernel.jsonl) and block size: `bin` set to
// it, building it on first use.
static int accelBinary(const char *dir, unsigned threads, char *bin, size_t binSize) {
  const char *lift = getenv("NVGPU_LEAN_LIFT");
  const char *nvcc = getenv("NVGPU_LEAN_ACCEL_NVCC") ? getenv("NVGPU_LEAN_ACCEL_NVCC") : "nvcc";
  const char *flags = getenv("NVGPU_LEAN_ACCEL_NVCC_FLAGS") ? getenv("NVGPU_LEAN_ACCEL_NVCC_FLAGS") : "";
  if (!lift) {
    note("NVGPU_LEAN_ACCEL needs NVGPU_LEAN_LIFT, the tools/lift directory");
    return 0;
  }
  char path[700];
  uint64_t h = fnv(accelArch(), strlen(accelArch()), 1469598103934665603ull);
  h = fnv(&threads, sizeof threads, h);
  snprintf(path, sizeof path, "%s/kernel.jsonl", dir);
  h = fnvFile(path, h);
  const char *sources[] = {"lift.py", "rt.cuh", "main.cu"};
  for (int i = 0; i < 3; i++) {
    snprintf(path, sizeof path, "%s/%s", lift, sources[i]);
    h = fnvFile(path, h);
  }
  char kdir[600];
  snprintf(kdir, sizeof kdir, "%s/accel/%016llx", cacheDir, (unsigned long long)h);
  snprintf(bin, binSize, "%s/lifted", kdir);
  struct stat st;
  if (stat(bin, &st) == 0) return 1;
  char cmd[4096], log[700];
  snprintf(cmd, sizeof cmd, "mkdir -p '%s'", kdir);
  if (system(cmd) != 0) {
    note("cannot create %s", kdir);
    return 0;
  }
  snprintf(log, sizeof log, "%s/build.log", kdir);
  snprintf(cmd, sizeof cmd,
           "{ cp '%s/mufu_tables.cuh' '%s/' 2>/dev/null || python3 '%s/lift.py' --tables '%s/mufu_tables.cuh'; } && "
           "python3 '%s/lift.py' --threads %u '%s/kernel.jsonl' > '%s/lifted.cu' && "
           "'%s' -std=c++17 -arch=%s -ftz=false -fmad=false %s -I'%s' -I'%s' -x cu -o '%s.tmp' '%s/main.cu' && "
           "mv '%s.tmp' '%s'",
           lift, kdir, lift, kdir, lift, threads, dir, kdir, nvcc, accelArch(), flags, kdir, lift, bin, lift, bin, bin);
  return accelRun("lifting and compiling the kernel", cmd, log);
}

// Launch `o` on the host's GPU (the lock held).
static CUresult accelKernel(Op *o) {
  CUfunction f = o->f;
  char dir[600];
  snprintf(dir, sizeof dir, "%s/accel", cacheDir);
  mkdirs(dir);
  snprintf(dir, sizeof dir, "%s/accel/run-XXXXXX", cacheDir);
  if (!mkdtemp(dir)) {
    note("cannot create a directory under %s/accel", cacheDir);
    return CUDA_ERROR_LAUNCH_FAILED;
  }
  CUdeviceptr lo = DEVICE_BASE, hi = DEVICE_BASE, mlo = DEVICE_BASE + G.part->memBytes, mhi = G.moduleNext;
  for (Alloc *a = G.allocs; a; a = a->next)
    if (a->base + a->size > hi) hi = a->base + a->size;
  char extra[256];
  snprintf(extra, sizeof extra, "globalBase %llu\nglobalSize %llu\nmoduleBase %llu\nmoduleSize %llu",
           (unsigned long long)lo, (unsigned long long)(hi - lo), (unsigned long long)mlo,
           (unsigned long long)(mhi - mlo));
  enterLean();
  lean_object *pb = lean_alloc_sarray(1, o->paramBytes, o->paramBytes);
  memcpy(lean_sarray_cptr(pb), o->params, o->paramBytes);
  lean_object *msg = ioValue(nvgpu_launch_export(lend(), (uint32_t)f->k[gcur], o->grid[0], o->grid[1],
                                                 o->grid[2], o->block[0], o->block[1], o->block[2],
                                                 o->cluster[0], o->cluster[1], o->cluster[2], o->smem,
                                                 pb, lean_mk_string(dir), lean_mk_string(extra),
                                                 lean_io_mk_world()));
  CUresult r = CUDA_ERROR_LAUNCH_FAILED;
  lean_object *g = NULL, *m = NULL;
  char path[700], bin[700], cmd[2400];
  if (!msg) goto done;
  if (lean_string_size(msg) != 1) {
    note("%s", lean_string_cstr(msg));
    lean_dec(msg);
    goto done;
  }
  lean_dec(msg);
  if (!(g = accelRead(lo, hi - lo)) || !(m = accelRead(mlo, mhi - mlo))) goto done;
  snprintf(path, sizeof path, "%s/global.bin", dir);
  if (!spill(path, lean_sarray_cptr(g), hi - lo)) goto done;
  snprintf(path, sizeof path, "%s/module.bin", dir);
  if (!spill(path, lean_sarray_cptr(m), mhi - mlo)) goto done;
  if (!accelBinary(dir, o->block[0] * o->block[1] * o->block[2], bin, sizeof bin)) goto done;
  const char *libs = getenv("NVGPU_LEAN_ACCEL_LIBRARY_PATH");
  snprintf(cmd, sizeof cmd, "env -u LD_PRELOAD LD_LIBRARY_PATH='%s' '%s' '%s' --out", libs ? libs : "",
           bin, dir);
  snprintf(path, sizeof path, "%s/run.log", dir);
  if (!accelRun("the lifted kernel", cmd, path)) goto done;
  size_t n;
  uint8_t *after;
  snprintf(path, sizeof path, "%s/out.bin", dir);
  if (!(after = (uint8_t *)slurp(path, &n)) || n != hi - lo) {
    free(after);
    note("the lifted kernel's global memory is missing or short");
    goto done;
  }
  int ok = accelTakeBack(lo, lean_sarray_cptr(g), after, n);
  free(after);
  if (!ok) goto done;
  snprintf(path, sizeof path, "%s/out-module.bin", dir);
  if (!(after = (uint8_t *)slurp(path, &n)) || n != mhi - mlo) {
    free(after);
    note("the lifted kernel's module memory is missing or short");
    goto done;
  }
  ok = accelTakeBack(mlo, lean_sarray_cptr(m), after, n);
  free(after);
  if (ok) r = CUDA_SUCCESS;
done:
  if (g) lean_dec(g);
  if (m) lean_dec(m);
  if (r == CUDA_SUCCESS) accelRemove(dir);
  return r;
}
