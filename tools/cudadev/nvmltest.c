// The management library of the emulated device, as nvidia-smi's default panel reads it: name,
// memory in use (published by the driver of other processes), processes, compute capability.
#include <nvml.h>
#include <stdio.h>

#define CK(x) do { nvmlReturn_t r_ = (x); if (r_ != NVML_SUCCESS) { \
  fprintf(stderr, "%s: %s\n", #x, nvmlErrorString(r_)); return 1; } } while (0)

int main(void) {
  CK(nvmlInit());
  unsigned n;
  CK(nvmlDeviceGetCount(&n));
  nvmlDevice_t d;
  CK(nvmlDeviceGetHandleByIndex(0, &d));
  char name[96];
  CK(nvmlDeviceGetName(d, name, sizeof name));
  nvmlMemory_t m;
  CK(nvmlDeviceGetMemoryInfo(d, &m));
  int major, minor;
  CK(nvmlDeviceGetCudaComputeCapability(d, &major, &minor));
  unsigned np = 8;
  nvmlProcessInfo_t ps[8];
  CK(nvmlDeviceGetComputeRunningProcesses(d, &np, ps));
  unsigned temp;
  nvmlReturn_t t = nvmlDeviceGetTemperature(d, NVML_TEMPERATURE_GPU, &temp);
  printf("%u device: %s, compute %d.%d, %llu of %llu MiB used, %u processes, temperature %s\n", n,
         name, major, minor, m.used >> 20, m.total >> 20, np,
         t == NVML_SUCCESS ? "?" : nvmlErrorString(t));
  return 0;
}
