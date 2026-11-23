// The accelerated mode's host side: run a lifted kernel (tools/lift/lift.py, compiled to a cubin)
// on one launch the simulator exported (`lake exe sim SPEC --export DIR`), compare its global
// memory afterwards with the simulator's byte for byte, and time it. Built once for every kernel:
// the kernel comes from its cubin (`lift_info`: the threads per CTA it was lifted for, whether it
// reads the clock, its parameter block's place in constant bank 0).
//
// Usage: lift-run CUBIN EXPORT_DIR [REPS] [--helpers N]
//   EXPORT_DIR holds meta.txt, const0.bin, global.bin (before) and final.bin (the simulator's after).
// Prints the mismatching bytes (0 means bit-exact), the first ones, and the kernel's time. A
// kernel that reads the clock and differs is CLOCK-DEPENDENT (exit 5), not a mismatch.
//   --helpers N: at most N helper CTAs (rt.cuh's work pool; default, every one the GPU holds
//   beside the launch's CTAs; 0, none).
//        lift-run CUBIN EXPORT_DIR --out
//   An accelerated backend's run: one run, its global memory after it written to out.bin, and to
//   out-module.bin the module segment's (meta.txt's moduleBase and moduleSize, module.bin before),
//   when there is one. Exits 4 on a fault.
//        lift-run --serve
//   The same runs for an accelerated backend, one process for all of them: a line `CUBIN DIR` on
//   stdin runs `CUBIN DIR --out` (its output to DIR/run.log) and answers `exit N`. Each cubin is
//   loaded once. Its device memory may instead stay on this GPU between runs (resident): lines
//     shm NAME BYTES        map the staging buffer the backend shares (POSIX shared memory)
//     seg S BASE SIZE       segment S (the backend's number) covers [BASE, BASE + SIZE),
//                           its bytes kept where the old and new ranges overlap, new bytes zero
//     put S OFF N           N bytes from the staging buffer to segment S at OFF
//     get S OFF N           N bytes of segment S at OFF to the staging buffer
//     run CUBIN DIR G M     run as above on resident segments G (allocations) and M (module data;
//                           DIR without global.bin and
//                           module.bin); the segments then hold the device's memory, which the
//                           backend reads back (`get`) only where it needs it
//   each answered `ok` (or `exit N` for a run; `fail` when a request cannot be met).
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// rt.cuh's types (`Ctx`, the work pool's) do not depend on the block size; its device functions
// are the kernel's, not this program's.
#define LIFT_THREADS 32
#include "rt.cuh"

static FILE* runLog = stdout;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(runLog, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)
#define CU(x) do { CUresult e_ = (x); if (e_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorString(e_, &s_); \
  fprintf(runLog, "CUDA %s at %s:%d\n", s_, __FILE__, __LINE__); return 1; } } while (0)

static bool readFile(const std::string& p, std::vector<uint8_t>* v) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(runLog, "cannot read %s\n", p.c_str()); return false; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  v->resize(n > 0 ? (size_t)n : 0);
  bool ok = v->empty() || fread(v->data(), 1, v->size(), f) == v->size();
  fclose(f);
  if (!ok) fprintf(runLog, "cannot read %s\n", p.c_str());
  return ok;
}

static std::map<std::string, std::vector<unsigned long long>> readMeta(const std::string& p) {
  std::map<std::string, std::vector<unsigned long long>> m;
  std::ifstream f(p);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream s(line);
    std::string k; s >> k;
    if (k == "kernel") continue;
    unsigned long long v;
    while (s >> v) m[k].push_back(v);
  }
  return m;
}

// Word `pairs[2i + 1]` at byte `pairs[2i]` of `base`, for every i below `n`.
__global__ void scatterWords(uint8_t* base, const uint32_t* pairs, uint32_t n) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    *(uint32_t*)(base + pairs[2 * i]) = pairs[2 * i + 1];
}

// A lifted kernel loaded from its cubin.
struct Kernel { CUfunction f; uint32_t threads, paramBase, paramBytes; bool clock; };

static int loadKernel(const std::string& cubin, Kernel* k) {
  CUmodule m;
  CU(cuModuleLoad(&m, cubin.c_str()));
  CU(cuModuleGetFunction(&k->f, m, "lifted"));
  CUdeviceptr info; size_t n;
  CU(cuModuleGetGlobal(&info, &n, m, "lift_info"));
  uint32_t v[4];
  CU(cuMemcpyDtoH(v, info, sizeof v));
  k->threads = v[0]; k->clock = v[1] != 0; k->paramBase = v[2]; k->paramBytes = v[3];
  return 0;
}

// Device memory freed when the run ends.
struct Arena {
  std::vector<void*> ps;
  ~Arena() { for (void* p : ps) cudaFree(p); }
  template <class T> cudaError_t alloc(T** p, size_t n) {
    cudaError_t e = cudaMalloc((void**)p, n ? n : 1);
    if (e == cudaSuccess) ps.push_back(*p);
    return e;
  }
};

// The resident segments (`--serve`'s `seg`): device memory kept between runs, at the addresses
// their bases name.
struct Segment { uint8_t* d = nullptr; uint64_t base = 0, size = 0; };
static std::map<unsigned, Segment> segs;
static double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

static int runDir(const Kernel& k, const std::string& dir, bool out1, int reps, long long helpersMax,
                  bool onDevice = false, unsigned gSeg = 0, unsigned mSeg = 1) {
  const auto t0 = std::chrono::steady_clock::now();
  double tRead = 0, tSetup = 0, tRun = 0;
  auto meta = readMeta(dir + "/meta.txt");
  const double tMeta = msSince(t0);
  std::vector<uint8_t> const0, global, module, final_;
  // `onDevice`: the segments are on the device already (`--serve`'s `run`).
  if (!readFile(dir + "/const0.bin", &const0) || (!onDevice && !readFile(dir + "/global.bin", &global))) return 1;
  bool hasModule = meta.count("moduleBase") > 0;
  if (hasModule && !onDevice && !readFile(dir + "/module.bin", &module)) return 1;
  if (!out1 && !readFile(dir + "/final.bin", &final_)) return 1;
  const double tFiles = msSince(t0) - tMeta;
  dim3 grid(meta["grid"][0], meta["grid"][1], meta["grid"][2]);
  dim3 block(meta["block"][0], meta["block"][1], meta["block"][2]);
  const uint32_t threads = block.x * block.y * block.z, warps = (threads + 31) / 32;
  if (threads != k.threads) { fprintf(runLog, "block size %u, lifted for %u\n", threads, k.threads); return 1; }
  uint32_t cx = meta["cluster"][0], cy = meta["cluster"][1], cz = meta["cluster"][2];
  uint32_t csize = cx * cy * cz;
  // Each CTA's shared window starts 1024-aligned, as a CTA's shared memory does (its accesses'
  // alignment is the kernel's to keep within it).
  uint64_t sharedBytes = (meta["shared"][0] + 1023) / 1024 * 1024, globalBase = meta["globalBase"][0];
  uint64_t ctas = (uint64_t)grid.x * grid.y * grid.z;

  // The device's properties, read once (the call takes milliseconds).
  static cudaDeviceProp prop;
  static bool haveProp = false;
  if (!haveProp) {
    int dev; CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    haveProp = true;
  }
  tRead = msSince(t0);
  fprintf(runLog, "device %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  Arena arena;
  uint8_t *dGlobal, *dShared, *dConst; uint32_t* dFault;
  if (onDevice) {
    if (segs[gSeg].base != globalBase || segs[gSeg].size < meta["globalSize"][0]) {
      fprintf(runLog, "the resident allocations do not cover the launch's\n");
      return 1;
    }
    dGlobal = segs[gSeg].d;
  } else {
    CK(arena.alloc(&dGlobal, global.size()));
    CK(cudaMemcpy(dGlobal, global.data(), global.size(), cudaMemcpyHostToDevice));
  }
  CK(arena.alloc(&dShared, sharedBytes * ctas));
  CK(arena.alloc(&dConst, const0.size() + 256));
  CK(arena.alloc(&dFault, 4));
  CK(cudaMemcpy(dConst, const0.data(), const0.size(), cudaMemcpyHostToDevice));

  lift::Ctx ctx{};
  ctx.sharedAll = dShared; ctx.sharedBytes = sharedBytes; ctx.const0 = dConst;
  ctx.globalDelta = (int64_t)((uint64_t)dGlobal - globalBase);
  ctx.hiBase = ~0ull;
  uint8_t* dModule = nullptr;
  if (hasModule && onDevice) {
    if (segs[mSeg].base != meta["moduleBase"][0] || segs[mSeg].size < meta["moduleSize"][0]) {
      fprintf(runLog, "the resident module data do not cover the launch's\n");
      return 1;
    }
    dModule = segs[mSeg].d;
    ctx.hiBase = meta["moduleBase"][0];
    ctx.hiDelta = (int64_t)((uint64_t)dModule - ctx.hiBase);
  } else if (hasModule) {
    CK(arena.alloc(&dModule, module.size() + 1));
    CK(cudaMemcpy(dModule, module.data(), module.size(), cudaMemcpyHostToDevice));
    ctx.hiBase = meta["moduleBase"][0];
    ctx.hiDelta = (int64_t)((uint64_t)dModule - ctx.hiBase);
  }
  ctx.paramBase = k.paramBase; ctx.paramBytes = k.paramBytes;
  ctx.clusterSize = csize; ctx.fault = dFault;
  ctx.cx = cx; ctx.cy = cy; ctx.cz = cz; ctx.gx = grid.x; ctx.gy = grid.y;
  // A cluster's CTAs wait on each other (its barrier, its peers' shared memory): the whole grid is
  // launched cooperatively, so every CTA is resident at once, or the run refuses.
  uint64_t clusters = ctas / csize;
  const size_t dyn = (size_t)warps * lift::WARP_SMEM;
  CU(cuFuncSetAttribute(k.f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)dyn));
  std::vector<unsigned long long> bars0(clusters, (unsigned long long)(csize * warps) << 20);
  unsigned long long* dBars; CK(arena.alloc(&dBars, 8 * clusters));
  ctx.clusterBars = dBars;
  uint8_t* dTmem; CK(arena.alloc(&dTmem, lift::TMEM_BYTES * ctas));
  ctx.tmemAll = dTmem;
  int perSM = 0;
  CU(cuOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, k.f, (int)threads, dyn));
  const uint64_t resident = (uint64_t)perSM * prop.multiProcessorCount;
  if (csize > 1 && resident < ctas) {
    fprintf(runLog, "a clustered grid of %llu CTAs does not fit at once (%d per SM)\n", (unsigned long long)ctas, perSM);
    return 1;
  }
  // The work pool's helpers: the CTAs the GPU holds beside the launch's, which then must all be
  // resident (a cooperative launch). Fewer helpers than the launch's CTAs serve its issuers one
  // after another and hold them up (the TMA + wgmma GEMM's 128 CTAs with 4 helpers on a GH200: 300
  // ms against 30 without), so a grid that leaves fewer idle runs without.
  uint64_t helpers = resident >= 2 * ctas ? resident - ctas : 0;
  if (helpersMax >= 0) helpers = std::min(helpers, (uint64_t)helpersMax);
  if (ctas + helpers > 0x7fffffffull) { fprintf(runLog, "a grid of %llu CTAs\n", (unsigned long long)(ctas + helpers)); return 1; }
  // Without helpers, a warp's copies are shared with its CTA's waiting warps through its slot
  // (`helpCta`), when the slots take at most 256 MB.
  const bool selfHelp = !helpers && ctas * warps * sizeof(lift::Slot) <= (256ull << 20);
  const uint64_t slots = helpers || selfHelp ? ctas * warps : 0;
  ctx.ctas = (uint32_t)ctas; ctx.helpers = (uint32_t)helpers; ctx.selfHelp = selfHelp;
  ctx.ringSize = (uint32_t)std::max<uint64_t>(2 * slots, 1);
  lift::Pool* dPool; CK(arena.alloc(&dPool, sizeof(lift::Pool)));
  lift::Slot* dSlots; CK(arena.alloc(&dSlots, sizeof(lift::Slot) * std::max<uint64_t>(slots, 1)));
  uint32_t* dAcc; CK(arena.alloc(&dAcc, 4ull * lift::ACC_WORDS * std::max<uint64_t>(slots, 1)));
  unsigned long long* dRing; CK(arena.alloc(&dRing, 8ull * ctx.ringSize));
  ctx.pool = dPool; ctx.slots = dSlots; ctx.acc = dAcc; ctx.ring = dRing;
  lift::Pool pool0{};
  pool0.live = (unsigned)(ctas * warps);
  const bool cooperative = csize > 1 || helpers > 0;
  fprintf(runLog, "%llu CTAs, %llu helper CTAs (%d per SM)\n", (unsigned long long)ctas, (unsigned long long)helpers, perSM);

  // Each CTA's reserved shared memory as the simulator starts it (`sharedInit<rank>`): its words
  // and their offsets, staged on the device once, written before every run (`scatterWords`).
  std::vector<uint32_t> init;
  for (uint64_t c = 0; c < ctas; ++c) {
    uint32_t x = c % grid.x, y = (c / grid.x) % grid.y, z = c / ((uint64_t)grid.x * grid.y);
    uint32_t rank = x % cx + cx * (y % cy + cy * (z % cz));
    auto it = meta.find("sharedInit" + std::to_string(rank));
    if (it == meta.end()) continue;
    for (size_t j = 0; j + 1 < it->second.size(); j += 2) {
      if (c * sharedBytes + it->second[j] > 0xfffffffcull) { fprintf(runLog, "shared windows past 4 GB\n"); return 1; }
      init.push_back((uint32_t)(c * sharedBytes + it->second[j]));
      init.push_back((uint32_t)it->second[j + 1]);
    }
  }
  uint32_t* dInit; CK(arena.alloc(&dInit, 4 * init.size()));
  CK(cudaMemcpy(dInit, init.data(), 4 * init.size(), cudaMemcpyHostToDevice));
  const uint32_t nInit = (uint32_t)(init.size() / 2);
  auto run = [&](float* ms) -> int {
    CK(cudaMemset(dShared, 0, sharedBytes * ctas));
    if (nInit) {
      scatterWords<<<(nInit + 255) / 256, 256>>>(dShared, dInit, nInit);
      CK(cudaGetLastError());
    }
    CK(cudaMemset(dFault, 0, 4));
    CK(cudaMemcpy(dBars, bars0.data(), 8 * clusters, cudaMemcpyHostToDevice));
    CK(cudaMemset(dTmem, 0, lift::TMEM_BYTES * ctas));
    // The pool empty: no job published (no ring entry's position matches), every slot at generation 0.
    CK(cudaMemcpy(dPool, &pool0, sizeof pool0, cudaMemcpyHostToDevice));
    CK(cudaMemset(dSlots, 0, sizeof(lift::Slot) * std::max<uint64_t>(slots, 1)));
    CK(cudaMemset(dRing, 0xff, 8ull * ctx.ringSize));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    void* args[] = {&ctx};
    const unsigned all = (unsigned)(ctas + helpers);
    if (cooperative) CU(cuLaunchCooperativeKernel(k.f, all, 1, 1, block.x, block.y, block.z, (unsigned)dyn, 0, args));
    else CU(cuLaunchKernel(k.f, all, 1, 1, block.x, block.y, block.z, (unsigned)dyn, 0, args, nullptr));
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return 0;
  };

  tSetup = msSince(t0) - tRead;
  float first;
  if (run(&first)) return 1;
  tRun = msSince(t0) - tRead - tSetup;
  uint32_t fault; CK(cudaMemcpy(&fault, dFault, 4, cudaMemcpyDeviceToHost));
  if (onDevice) {
    fprintf(runLog, "fault flags %#x, kernel time %.3f ms\n", fault, first);
    fprintf(runLog, "phases: read %.1f ms (meta %.1f, files %.1f of %zu bytes), setup %.1f ms, run %.1f ms, out %.1f ms\n",
            tRead, tMeta, tFiles, const0.size(), tSetup, tRun, msSince(t0) - tRead - tSetup - tRun);
    return fault == 0 ? 0 : 4;
  }
  std::vector<uint8_t> out(global.size());
  CK(cudaMemcpy(out.data(), dGlobal, out.size(), cudaMemcpyDeviceToHost));
  if (out1) {
    auto save = [&](const std::string& name, const std::vector<uint8_t>& h) {
      std::ofstream f(dir + "/" + name, std::ios::binary);
      f.write((const char*)h.data(), (std::streamsize)h.size());
      if (!f) { fprintf(runLog, "cannot write %s\n", name.c_str()); return false; }
      return true;
    };
    if (!save("out.bin", out)) return 1;
    if (hasModule) {
      std::vector<uint8_t> m(module.size());
      if (!m.empty()) CK(cudaMemcpy(m.data(), dModule, m.size(), cudaMemcpyDeviceToHost));
      if (!save("out-module.bin", m)) return 1;
    }
    fprintf(runLog, "fault flags %#x, kernel time %.3f ms\n", fault, first);
    fprintf(runLog, "phases: read %.1f ms (meta %.1f, files %.1f of %zu bytes), setup %.1f ms, run %.1f ms, out %.1f ms\n",
            tRead, tMeta, tFiles, const0.size() + global.size() + module.size(), tSetup, tRun,
            msSince(t0) - tRead - tSetup - tRun);
    return fault == 0 ? 0 : 4;
  }
  size_t n = std::min(out.size(), final_.size()), bad = 0;
  std::vector<size_t> where;
  for (size_t i = 0; i < n; ++i) if (out[i] != final_[i]) { if (where.size() < 8) where.push_back(i); ++bad; }
  size_t changed = 0;
  for (size_t i = 0; i < n; ++i) if (final_[i] != global[i]) ++changed;
  fprintf(runLog, "fault flags %#x\n", fault);
  fprintf(runLog, "bytes the simulator's run changed: %zu; bytes differing from the simulator's result: %zu of %zu\n", changed, bad, n);
  for (size_t i : where) fprintf(runLog, "  differs at +%#zx (address %#llx): lifted %02x, simulator %02x, before %02x\n",
                                 i, (unsigned long long)(globalBase + i), out[i], final_[i], global[i]);

  std::vector<float> t;
  for (int r = 0; r < reps; ++r) { float ms; if (run(&ms)) return 1; t.push_back(ms); }
  std::sort(t.begin(), t.end());
  if (!t.empty()) fprintf(runLog, "kernel time: first %.3f ms, median of %d %.3f ms, min %.3f ms\n", first, reps, t[t.size() / 2], t[0]);
  const bool exact = bad == 0 && fault == 0;
  fprintf(runLog, "%s\n", exact ? "BIT-EXACT" : k.clock && fault == 0 ? "CLOCK-DEPENDENT" : "MISMATCH");
  return exact ? 0 : k.clock && fault == 0 ? 5 : 3;
}

int main(int argc, char** argv) {
  CK(cudaFree(nullptr));  // the primary context, which the driver calls below share
  if (argc == 2 && std::string(argv[1]) == "--serve") {
    std::map<std::string, Kernel> kernels;
    uint8_t* stage = nullptr;
    uint64_t stageBytes = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
      std::istringstream in(line);
      std::string cmd;
      in >> cmd;
      // The staging buffer the backend shares: a fixed window that `put` and `get` move bytes
      // through, never a whole segment.
      if (cmd == "shm") {
        std::string name; in >> name >> stageBytes;
        int fd = shm_open(name.c_str(), O_RDWR, 0);
        void* p = fd < 0 ? MAP_FAILED : mmap(nullptr, stageBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (fd >= 0) close(fd);
        stage = p == MAP_FAILED ? nullptr : (uint8_t*)p;
        printf(stage ? "ok\n" : "fail\n");
        fflush(stdout);
        continue;
      }
      if (cmd == "seg") {
        unsigned i; uint64_t base, size; in >> i >> base >> size;
        bool ok = true;
        if (segs[i].base != base || segs[i].size != size) {
          // A new range: the bytes where the old and the new overlap kept, the rest zero.
          uint8_t* d = nullptr;
          ok = cudaMalloc(&d, size ? size : 1) == cudaSuccess && cudaMemset(d, 0, size ? size : 1) == cudaSuccess;
          const uint64_t lo = std::max(base, segs[i].base), hi = std::min(base + size, segs[i].base + segs[i].size);
          if (ok && segs[i].d && lo < hi)
            ok = cudaMemcpy(d + (lo - base), segs[i].d + (lo - segs[i].base), hi - lo, cudaMemcpyDeviceToDevice) == cudaSuccess;
          if (ok) {
            if (segs[i].d) cudaFree(segs[i].d);
            segs[i].d = d; segs[i].base = base; segs[i].size = size;
          } else if (d) cudaFree(d);
        }
        printf(ok ? "ok\n" : "fail\n");
        fflush(stdout);
        continue;
      }
      if (cmd == "put" || cmd == "get") {
        unsigned i; uint64_t off, n; in >> i >> off >> n;
        bool ok = stage && segs.count(i) && n <= stageBytes && off + n <= segs[i].size;
        if (ok && n) ok = (cmd == "put" ? cudaMemcpy(segs[i].d + off, stage, n, cudaMemcpyHostToDevice)
                                        : cudaMemcpy(stage, segs[i].d + off, n, cudaMemcpyDeviceToHost)) == cudaSuccess;
        printf(ok ? "ok\n" : "fail\n");
        fflush(stdout);
        continue;
      }
      // `run CUBIN DIR` on the resident segments, or `CUBIN DIR` from the files in DIR.
      const bool resident = cmd == "run";
      std::string cubin = resident ? "" : cmd, dir;
      unsigned gSeg = 0, mSeg = 1;
      if (resident) in >> cubin;
      in >> dir;
      if (resident) in >> gSeg >> mSeg;
      FILE* f = fopen((dir + "/run.log").c_str(), "w");
      runLog = f ? f : stderr;
      int code = 0;
      auto it = kernels.find(cubin);
      if (it == kernels.end()) {
        Kernel k;
        code = loadKernel(cubin, &k);
        if (code == 0) it = kernels.emplace(cubin, k).first;
      }
      if (code == 0) code = runDir(it->second, dir, true, 0, -1, resident, gSeg, mSeg);
      if (f) fclose(f);
      runLog = stdout;
      printf("exit %d\n", code);
      fflush(stdout);
      // A failed CUDA call may have left the context unusable: the next request starts afresh.
      if (cudaGetLastError() != cudaSuccess) return 1;
    }
    return 0;
  }
  if (argc < 3) { fprintf(stderr, "usage: lift-run CUBIN EXPORT_DIR [REPS | --out] [--helpers N] | --serve\n"); return 2; }
  bool out1 = false;
  int reps = 20;
  long long helpersMax = -1;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--out") out1 = true;
    else if (a == "--helpers" && i + 1 < argc) helpersMax = atoll(argv[++i]);
    else reps = atoi(argv[i]);
  }
  Kernel k;
  if (loadKernel(argv[1], &k)) return 1;
  return runDir(k, argv[2], out1, reps, helpersMax);
}
