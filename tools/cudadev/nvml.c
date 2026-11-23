// libnvidia-ml.so.1 for the emulated device: NVML over what the emulated driver publishes
// (state.h). Unproven glue. The name nvidia-smi and NVML clients show is the part's, followed by
// nothing that says "emulated": the PCI device ID and serial ("nvgpu-lean") do. The devices are the parts NVGPU_LEAN_GPU names (default h200; `b200x8` eight); memory
// in use, processes and utilization come from the driver's record. What an emulated device has no
// value for (temperature, fans, power draw, ECC counts, clocks it does not vary) is reported as
// not supported, which nvidia-smi shows as N/A. Every other NVML function returns
// NVML_ERROR_NOT_SUPPORTED (nvml_stubs.h, generated from nvml.h), and the deprecated unversioned
// names are there too, as NVIDIA's library has them: those taking what their successors take
// (`nvmlInit`, `nvmlDeviceGetHandleByIndex`) call them.

#include <nvml.h>
#include <stdlib.h>
#include <string.h>

#include "parts.h"
#include "state.h"

#define MAX_DEVICES 16
static State *state;
static struct nvmlDevice_st {
  const Part *part;
  unsigned index;
} devices[MAX_DEVICES];
static unsigned ndevices;

nvmlReturn_t DECLDIR nvmlInit_v2(void) {
  const Part *ps[MAX_DEVICES];
  int n = partsOf(getenv("NVGPU_LEAN_GPU"), ps, MAX_DEVICES);
  if (n == 0) return NVML_ERROR_DRIVER_NOT_LOADED;
  for (int i = 0; i < n; i++) devices[i] = (struct nvmlDevice_st){ps[i], (unsigned)i};
  ndevices = (unsigned)n;
  state = stateMap(0);
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlInitWithFlags(unsigned int flags) {
  (void)flags;
  return nvmlInit_v2();
}
nvmlReturn_t DECLDIR nvmlShutdown(void) { return NVML_SUCCESS; }

const char *DECLDIR nvmlErrorString(nvmlReturn_t r) {
  switch (r) {
    case NVML_SUCCESS: return "Success";
    case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
    case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
    case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
    default: return "Unknown Error";
  }
}

#define NEED() \
  do { if (!ndevices) return NVML_ERROR_UNINITIALIZED; } while (0)
// A device argument checked; `part` and `idx` its own.
#define DEV(d) \
  NEED(); \
  if ((d) < devices || (d) >= devices + ndevices) return NVML_ERROR_INVALID_ARGUMENT; \
  const Part *part = (d)->part; \
  const unsigned idx = (d)->index; \
  (void)part; (void)idx

static nvmlReturn_t copy(char *dst, unsigned int len, const char *src) {
  if (strlen(src) + 1 > len) return NVML_ERROR_INSUFFICIENT_SIZE;
  strcpy(dst, src);
  return NVML_SUCCESS;
}

// The driver branch the device reports, which tools compare with their own (nvidia-smi requires
// its own version): NVGPU_LEAN_DRIVER_VERSION, by default CUDA 13.0's driver.
static const char *driverVersion(void) {
  const char *v = getenv("NVGPU_LEAN_DRIVER_VERSION");
  return v ? v : "580.65.06";
}
nvmlReturn_t DECLDIR nvmlSystemGetDriverVersion(char *v, unsigned int len) {
  return copy(v, len, driverVersion());
}
nvmlReturn_t DECLDIR nvmlSystemGetNVMLVersion(char *v, unsigned int len) {
  return copy(v, len, driverVersion());
}
nvmlReturn_t DECLDIR nvmlSystemGetCudaDriverVersion(int *v) {
  *v = 13000;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlSystemGetCudaDriverVersion_v2(int *v) { return nvmlSystemGetCudaDriverVersion(v); }

nvmlReturn_t DECLDIR nvmlDeviceGetCount_v2(unsigned int *n) {
  NEED();
  *n = ndevices;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByIndex_v2(unsigned int i, nvmlDevice_t *d) {
  NEED();
  if (i >= ndevices) return NVML_ERROR_INVALID_ARGUMENT;
  *d = &devices[i];
  return NVML_SUCCESS;
}
// Device i's UUID: a fixed prefix and the index.
static void uuidOf(unsigned i, char *out, size_t len) {
  snprintf(out, len, "GPU-6e766770-752d-6c65-616e-2d656d75%04x", i);
}
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *d) {
  NEED();
  for (unsigned i = 0; i < ndevices; i++) {
    char u[96];
    uuidOf(i, u, sizeof u);
    if (uuid && strcmp(uuid, u) == 0) return nvmlDeviceGetHandleByIndex_v2(i, d);
  }
  return NVML_ERROR_NOT_FOUND;
}
nvmlReturn_t DECLDIR nvmlDeviceGetHandleByPciBusId_v2(const char *id, nvmlDevice_t *d) {
  NEED();
  unsigned dom = 0, bus = 0;
  if (!id || sscanf(id, "%x:%x", &dom, &bus) != 2) return NVML_ERROR_INVALID_ARGUMENT;
  for (unsigned i = 0; i < ndevices; i++)
    if ((unsigned)devices[i].part->pciBus + i == bus) return nvmlDeviceGetHandleByIndex_v2(i, d);
  return NVML_ERROR_NOT_FOUND;
}
nvmlReturn_t DECLDIR nvmlDeviceGetIndex(nvmlDevice_t d, unsigned int *i) {
  DEV(d);
  *i = idx;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetMinorNumber(nvmlDevice_t d, unsigned int *i) {
  DEV(d);
  *i = idx;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetName(nvmlDevice_t d, char *name, unsigned int len) {
  DEV(d);
  return copy(name, len, part->marketing);
}
nvmlReturn_t DECLDIR nvmlDeviceGetBrand(nvmlDevice_t d, nvmlBrandType_t *b) {
  DEV(d);
  *b = NVML_BRAND_NVIDIA;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetUUID(nvmlDevice_t d, char *uuid, unsigned int len) {
  DEV(d);
  char u[96];
  uuidOf(idx, u, sizeof u);
  return copy(uuid, len, u);
}
nvmlReturn_t DECLDIR nvmlDeviceGetSerial(nvmlDevice_t d, char *s, unsigned int len) {
  DEV(d);
  return copy(s, len, "nvgpu-lean");
}
nvmlReturn_t DECLDIR nvmlDeviceGetPciInfo_v3(nvmlDevice_t d, nvmlPciInfo_t *p) {
  DEV(d);
  memset(p, 0, sizeof *p);
  p->bus = part->pciBus + idx;
  p->pciDeviceId = 0x10de;
  snprintf(p->busIdLegacy, sizeof p->busIdLegacy, "0000:%02X:00.0", part->pciBus + idx);
  snprintf(p->busId, sizeof p->busId, "00000000:%02X:00.0", part->pciBus + idx);
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetCudaComputeCapability(nvmlDevice_t d, int *major, int *minor) {
  DEV(d);
  *major = part->ccMajor;
  *minor = part->ccMinor;
  return NVML_SUCCESS;
}

// Memory in use: the record keeps the total over the devices, reported on the first.
static uint64_t used(unsigned i) { return i == 0 && state && state->magic == STATE_MAGIC ? state->memUsed : 0; }

nvmlReturn_t DECLDIR nvmlDeviceGetMemoryInfo(nvmlDevice_t d, nvmlMemory_t *m) {
  DEV(d);
  m->total = part->memBytes;
  m->used = used(idx);
  m->free = m->total - m->used;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t d, nvmlMemory_v2_t *m) {
  DEV(d);
  m->total = part->memBytes;
  m->reserved = 0;
  m->used = used(idx);
  m->free = m->total - m->used;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetUtilizationRates(nvmlDevice_t d, nvmlUtilization_t *u) {
  DEV(d);
  int busy = state && state->magic == STATE_MAGIC && state->busy;
  u->gpu = busy ? 100 : 0;
  u->memory = 0;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetComputeMode(nvmlDevice_t d, nvmlComputeMode_t *m) {
  DEV(d);
  *m = NVML_COMPUTEMODE_DEFAULT;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetPersistenceMode(nvmlDevice_t d, nvmlEnableState_t *m) {
  DEV(d);
  *m = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetDisplayMode(nvmlDevice_t d, nvmlEnableState_t *m) {
  DEV(d);
  *m = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetDisplayActive(nvmlDevice_t d, nvmlEnableState_t *m) {
  DEV(d);
  *m = NVML_FEATURE_DISABLED;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetPerformanceState(nvmlDevice_t d, nvmlPstates_t *p) {
  DEV(d);
  *p = NVML_PSTATE_0;
  return NVML_SUCCESS;
}
// The part's clock, which the model holds fixed (its boost clock, `clockMHz`).
nvmlReturn_t DECLDIR nvmlDeviceGetClockInfo(nvmlDevice_t d, nvmlClockType_t t, unsigned int *mhz) {
  DEV(d);
  if (t != NVML_CLOCK_SM && t != NVML_CLOCK_GRAPHICS) return NVML_ERROR_NOT_SUPPORTED;
  *mhz = part->clockMHz;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetMaxClockInfo(nvmlDevice_t d, nvmlClockType_t t, unsigned int *mhz) {
  return nvmlDeviceGetClockInfo(d, t, mhz);
}
nvmlReturn_t DECLDIR nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t d, unsigned int *n,
                                                            nvmlProcessInfo_t *ps) {
  DEV(d);
  unsigned k = 0;
  if (state && state->magic == STATE_MAGIC)
    for (int i = 0; i < STATE_PROCS; i++)
      if (state->procs[i].pid) {
        if (ps && k < *n) {
          memset(&ps[k], 0, sizeof ps[k]);
          ps[k].pid = state->procs[i].pid;
          ps[k].usedGpuMemory = state->procs[i].memUsed;
          ps[k].gpuInstanceId = 0xFFFFFFFF;
          ps[k].computeInstanceId = 0xFFFFFFFF;
        }
        k++;
      }
  nvmlReturn_t r = ps && k > *n ? NVML_ERROR_INSUFFICIENT_SIZE : NVML_SUCCESS;
  *n = k;
  return r;
}
nvmlReturn_t DECLDIR nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t d, unsigned int *n,
                                                             nvmlProcessInfo_t *ps) {
  DEV(d);
  (void)ps;
  *n = 0;
  return NVML_SUCCESS;
}
nvmlReturn_t DECLDIR nvmlDeviceGetMigMode(nvmlDevice_t d, unsigned int *cur, unsigned int *pend) {
  DEV(d);
  *cur = *pend = NVML_DEVICE_MIG_DISABLE;
  return NVML_SUCCESS;
}

#include "nvml_stubs.h"
