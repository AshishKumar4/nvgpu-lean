// The management library of the emulated device, as nvidia-smi's default panel reads it: name,
// memory in use (published by the driver of other processes), processes, compute capability. And
// the deprecated unversioned names, as libraries find them by name (NCCL's and NVSHMEM's
// `dlsym(h, "nvmlInit")`): each present and answering as its successor does.
#include <dlfcn.h>
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
  void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  nvmlReturn_t (*init)(void) = lib ? (nvmlReturn_t (*)(void))dlsym(lib, "nvmlInit") : NULL;
  nvmlReturn_t (*count)(unsigned *) = lib ? (nvmlReturn_t (*)(unsigned *))dlsym(lib, "nvmlDeviceGetCount") : NULL;
  nvmlReturn_t (*handle)(unsigned, nvmlDevice_t *) =
      lib ? (nvmlReturn_t (*)(unsigned, nvmlDevice_t *))dlsym(lib, "nvmlDeviceGetHandleByIndex") : NULL;
  nvmlReturn_t (*byBus)(const char *, nvmlDevice_t *) =
      lib ? (nvmlReturn_t (*)(const char *, nvmlDevice_t *))dlsym(lib, "nvmlDeviceGetHandleByPciBusId") : NULL;
  if (!init || !count || !handle || !byBus) {
    fprintf(stderr, "an unversioned NVML name is missing\n");
    return 1;
  }
  unsigned n1;
  nvmlDevice_t d1;
  nvmlPciInfo_t pci;
  CK(init());
  CK(count(&n1));
  CK(handle(0, &d1));
  CK(nvmlDeviceGetPciInfo(d1, &pci));
  nvmlDevice_t d2;
  CK(byBus(pci.busId, &d2));
  if (n1 != n || d1 != d || d2 != d) {
    fprintf(stderr, "the unversioned NVML names answer differently\n");
    return 1;
  }
  printf("unversioned names: %u devices, device 0 by index and by bus id\n", n1);
  unsigned gen, width;
  CK(nvmlDeviceGetCurrPcieLinkGeneration(d, &gen));
  CK(nvmlDeviceGetCurrPcieLinkWidth(d, &width));
  if ((gen != 4 && gen != 5) || width != 16) {
    fprintf(stderr, "PCIe link %u x%u\n", gen, width);
    return 1;
  }
  printf("PCIe %u.0 x%u\n", gen, width);
  return 0;
}
