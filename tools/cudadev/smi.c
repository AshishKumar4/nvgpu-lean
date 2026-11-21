// nvidia-smi for the emulated device, over the public NVML API (NVIDIA's nvidia-smi does not run
// on the emulated NVML). It prints the default panel, `-L`,
// `-q` and `--query-gpu=FIELDS --format=csv[,noheader][,nounits]` for the fields scripts read.
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static nvmlDevice_t dev;  // the device being printed
static unsigned ndev, idx;

static void na(char *buf, size_t n, nvmlReturn_t r, const char *fmt, unsigned long long v) {
  if (r == NVML_SUCCESS) snprintf(buf, n, fmt, v);
  else snprintf(buf, n, "N/A");
}

// One --query-gpu field, or NULL if unknown.
static const char *field(const char *f, int units, char *buf, size_t n) {
  nvmlMemory_t m;
  nvmlUtilization_t u;
  unsigned v = 0;
  if (!strcmp(f, "name") || !strcmp(f, "gpu_name")) {
    nvmlDeviceGetName(dev, buf, n);
  } else if (!strcmp(f, "index")) {
    snprintf(buf, n, "%u", idx);
  } else if (!strcmp(f, "uuid") || !strcmp(f, "gpu_uuid")) {
    nvmlDeviceGetUUID(dev, buf, n);
  } else if (!strcmp(f, "driver_version")) {
    nvmlSystemGetDriverVersion(buf, n);
  } else if (!strcmp(f, "pci.bus_id")) {
    nvmlPciInfo_t p;
    nvmlDeviceGetPciInfo(dev, &p);
    snprintf(buf, n, "%s", p.busId);
  } else if (!strcmp(f, "memory.total") || !strcmp(f, "memory.used") || !strcmp(f, "memory.free")) {
    nvmlDeviceGetMemoryInfo(dev, &m);
    unsigned long long x = f[7] == 't' ? m.total : f[7] == 'u' ? m.used : m.free;
    snprintf(buf, n, units ? "%llu MiB" : "%llu", x >> 20);
  } else if (!strcmp(f, "utilization.gpu") || !strcmp(f, "utilization.memory")) {
    nvmlDeviceGetUtilizationRates(dev, &u);
    snprintf(buf, n, units ? "%u %%" : "%u", f[12] == 'g' ? u.gpu : u.memory);
  } else if (!strcmp(f, "clocks.sm") || !strcmp(f, "clocks.max.sm") || !strcmp(f, "clocks.gr")) {
    na(buf, n, nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &v), units ? "%llu MHz" : "%llu", v);
  } else if (!strcmp(f, "compute_cap")) {
    int a, b;
    nvmlDeviceGetCudaComputeCapability(dev, &a, &b);
    snprintf(buf, n, "%d.%d", a, b);
  } else if (!strcmp(f, "temperature.gpu") || !strcmp(f, "power.draw") ||
             !strcmp(f, "fan.speed") || !strcmp(f, "power.limit")) {
    snprintf(buf, n, "[N/A]");
  } else if (!strcmp(f, "count")) {
    snprintf(buf, n, "1");
  } else {
    return NULL;
  }
  return buf;
}

static int query(const char *fields, const char *format) {
  int header = !strstr(format, "noheader"), units = !strstr(format, "nounits");
  char list[1024];
  snprintf(list, sizeof list, "%s", fields);
  char *names[64];
  int k = 0;
  for (char *t = strtok(list, ","); t && k < 64; t = strtok(NULL, ",")) {
    while (*t == ' ') t++;
    names[k++] = t;
  }
  char buf[256];
  if (header)
    for (int i = 0; i < k; i++) printf("%s%s", names[i], i + 1 < k ? ", " : "\n");
  for (idx = 0; idx < ndev; idx++) {
    nvmlDeviceGetHandleByIndex(idx, &dev);
    for (int i = 0; i < k; i++) {
      const char *v = field(names[i], units, buf, sizeof buf);
      if (!v) {
        fprintf(stderr, "Field \"%s\" is not a valid field to query.\n", names[i]);
        return 2;
      }
      printf("%s%s", v, i + 1 < k ? ", " : "\n");
    }
  }
  return 0;
}

static void panel(void) {
  char name[96], drv[64], buf[64];
  nvmlSystemGetDriverVersion(drv, sizeof drv);
  int cuda;
  nvmlSystemGetCudaDriverVersion(&cuda);
  time_t now = time(NULL);
  char when[64];
  strftime(when, sizeof when, "%a %b %e %H:%M:%S %Y", localtime(&now));
  printf("%s\n", when);
  printf("+-----------------------------------------------------------------------------------------+\n");
  snprintf(buf, sizeof buf, "%d.%d", cuda / 1000, cuda % 1000 / 10);
  char line[256];
  snprintf(line, sizeof line, "| NVIDIA-SMI (nvgpu-lean)   Driver Version: %s      CUDA Version: %s", drv, buf);
  printf("%-90s|\n", line);
  printf("|-----------------------------------------+------------------------+----------------------+\n");
  printf("| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n");
  printf("| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n");
  printf("|=========================================+========================+======================|\n");
  for (unsigned i = 0; i < ndev; i++) {
    nvmlDeviceGetHandleByIndex(i, &dev);
    nvmlDeviceGetName(dev, name, sizeof name);
    nvmlMemory_t m;
    nvmlDeviceGetMemoryInfo(dev, &m);
    nvmlUtilization_t u;
    nvmlDeviceGetUtilizationRates(dev, &u);
    nvmlPciInfo_t p;
    nvmlDeviceGetPciInfo(dev, &p);
    printf("| %3u  %-24.24s %9s | %-16s %5s | %20s |\n", i, name, "Off", p.busId + 4, "Off", "N/A");
    char mem[64];
    snprintf(mem, sizeof mem, "%lluMiB / %lluMiB", m.used >> 20, m.total >> 20);
    printf("| N/A   N/A    P0             N/A /  N/A  | %22s | %6u%%      Default |\n", mem, u.gpu);
    printf("+-----------------------------------------+------------------------+----------------------+\n");
  }
  printf("\n");
  printf("+-----------------------------------------------------------------------------------------+\n");
  printf("| Processes:                                                                              |\n");
  printf("|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n");
  printf("|=========================================================================================|\n");
  unsigned n = 16;
  nvmlProcessInfo_t ps[16];
  nvmlDeviceGetHandleByIndex(0, &dev);
  if (nvmlDeviceGetComputeRunningProcesses(dev, &n, ps) != NVML_SUCCESS || n == 0)
    printf("|  No running processes found                                                             |\n");
  else
    for (unsigned i = 0; i < n; i++)
      printf("|  %3d  N/A  N/A   %14u      C   %-35s %7lluMiB |\n", 0, ps[i].pid, "",
             ps[i].usedGpuMemory >> 20);
  printf("+-----------------------------------------------------------------------------------------+\n");
}

int main(int argc, char **argv) {
  if (nvmlInit() != NVML_SUCCESS || nvmlDeviceGetCount(&ndev) != NVML_SUCCESS || ndev == 0 ||
      nvmlDeviceGetHandleByIndex(0, &dev) != NVML_SUCCESS) {
    fprintf(stderr, "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n");
    return 9;
  }
  const char *q = NULL, *fmt = "csv";
  for (int i = 1; i < argc; i++) {
    if (!strncmp(argv[i], "--query-gpu=", 12)) q = argv[i] + 12;
    else if (!strncmp(argv[i], "--format=", 9)) fmt = argv[i] + 9;
    else if (!strcmp(argv[i], "-L") || !strcmp(argv[i], "--list-gpus")) {
      for (unsigned i = 0; i < ndev; i++) {
        char name[96], uuid[96];
        nvmlDeviceGetHandleByIndex(i, &dev);
        nvmlDeviceGetName(dev, name, sizeof name);
        nvmlDeviceGetUUID(dev, uuid, sizeof uuid);
        printf("GPU %u: %s (UUID: %s)\n", i, name, uuid);
      }
      return 0;
    } else if (!strcmp(argv[i], "-q")) {
      char buf[256];
      const char *fs[] = {"name", "uuid", "driver_version", "pci.bus_id", "compute_cap",
                          "memory.total", "memory.used", "memory.free", "utilization.gpu",
                          "clocks.sm"};
      printf("\n==============NVSMI LOG==============\n\nAttached GPUs : %u\n", ndev);
      for (idx = 0; idx < ndev; idx++) {
        nvmlDeviceGetHandleByIndex(idx, &dev);
        printf("GPU %u\n", idx);
        for (unsigned k = 0; k < sizeof fs / sizeof fs[0]; k++)
          printf("    %-30s : %s\n", fs[k], field(fs[k], 1, buf, sizeof buf));
      }
      return 0;
    }
  }
  if (q) return query(q, fmt);
  panel();
  return 0;
}
