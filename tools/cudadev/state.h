// What the emulated driver (libcuda.so.1) publishes for the management library (libnvidia-ml.so.1)
// of other processes, as a real driver's kernel module would: a shared-memory record per user.
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define STATE_PROCS 16

typedef struct {
  uint32_t pid;
  uint64_t memUsed;
} StateProc;

typedef struct {
  uint32_t magic;         // 'NVGL'
  char partName[32];      // the emulated parts, as NVGPU_LEAN_GPU names them
  uint64_t memUsed;       // device memory allocated, over every process
  uint64_t launches;      // kernels launched
  uint32_t busy;          // a launch is running
  StateProc procs[STATE_PROCS];
} State;

#define STATE_MAGIC 0x4c47564eu

// The record, mapped (created by the driver side); NULL if unavailable.
static State *stateMap(int create) {
  char path[64];
  snprintf(path, sizeof path, "/dev/shm/nvgpu-lean-%u", (unsigned)getuid());
  int fd = open(path, create ? O_RDWR | O_CREAT : O_RDONLY, 0600);
  if (fd < 0) return NULL;
  if (create && ftruncate(fd, sizeof(State)) != 0) {
    close(fd);
    return NULL;
  }
  void *p = mmap(NULL, sizeof(State), create ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  return p == MAP_FAILED ? NULL : p;
}
