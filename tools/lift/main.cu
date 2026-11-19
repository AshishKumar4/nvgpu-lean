// The accelerated mode's host side: run a lifted kernel (tools/lift/lift.py) on one launch the
// simulator exported (`lake exe sim SPEC --export DIR`), compare its global memory afterwards with
// the simulator's byte for byte, and time it.
//
// Usage: lifted EXPORT_DIR [REPS]
//   EXPORT_DIR holds meta.txt, const0.bin, global.bin (before) and final.bin (the simulator's after).
// Prints the mismatching bytes (0 means bit-exact), the first ones, and the kernel's time.
//        lifted EXPORT_DIR --out
//   An accelerated backend's run: one run, its global memory
//   after it written to out.bin, and to out-module.bin the module segment's (meta.txt's moduleBase
//   and moduleSize, module.bin before), when there is one. Exits 4 on a fault.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "lifted.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static std::vector<uint8_t> readFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
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

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: lifted EXPORT_DIR [REPS | --out]\n"); return 2; }
  std::string dir = argv[1];
  bool out1 = argc > 2 && std::string(argv[2]) == "--out";
  int reps = argc > 2 && !out1 ? atoi(argv[2]) : 20;
  auto meta = readMeta(dir + "/meta.txt");
  auto const0 = readFile(dir + "/const0.bin");
  auto global = readFile(dir + "/global.bin");
  bool hasModule = meta.count("moduleBase") > 0;
  auto module = hasModule ? readFile(dir + "/module.bin") : std::vector<uint8_t>();
  auto final_ = out1 ? std::vector<uint8_t>() : readFile(dir + "/final.bin");
  dim3 grid(meta["grid"][0], meta["grid"][1], meta["grid"][2]);
  dim3 block(meta["block"][0], meta["block"][1], meta["block"][2]);
  if (block.x * block.y * block.z != LIFT_THREADS) { fprintf(stderr, "block size %u, lifted for %d\n",
      block.x * block.y * block.z, LIFT_THREADS); return 1; }
  uint32_t cx = meta["cluster"][0], cy = meta["cluster"][1], cz = meta["cluster"][2];
  uint32_t csize = cx * cy * cz;
  // Each CTA's shared window starts 1024-aligned, as a CTA's shared memory does (its accesses'
  // alignment is the kernel's to keep within it).
  uint64_t sharedBytes = (meta["shared"][0] + 1023) / 1024 * 1024, globalBase = meta["globalBase"][0];
  uint64_t ctas = (uint64_t)grid.x * grid.y * grid.z;

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("device %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  uint8_t *dGlobal, *dShared, *dConst; uint32_t* dFault;
  CK(cudaMalloc(&dGlobal, global.size()));
  CK(cudaMalloc(&dShared, sharedBytes * ctas));
  CK(cudaMalloc(&dConst, const0.size() + 256));
  CK(cudaMalloc(&dFault, 4));
  CK(cudaMemcpy(dGlobal, global.data(), global.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dConst, const0.data(), const0.size(), cudaMemcpyHostToDevice));

  lift::Ctx ctx{};
  ctx.sharedAll = dShared; ctx.sharedBytes = sharedBytes; ctx.const0 = dConst;
  ctx.globalDelta = (int64_t)((uint64_t)dGlobal - globalBase);
  ctx.hiBase = ~0ull;
  uint8_t* dModule = nullptr;
  if (hasModule) {
    CK(cudaMalloc(&dModule, module.size() + 1));
    CK(cudaMemcpy(dModule, module.data(), module.size(), cudaMemcpyHostToDevice));
    ctx.hiBase = meta["moduleBase"][0];
    ctx.hiDelta = (int64_t)((uint64_t)dModule - ctx.hiBase);
  }
  ctx.paramBase = PARAM_BASE; ctx.paramBytes = PARAM_BYTES; ctx.clusterSize = csize; ctx.fault = dFault;
  ctx.cx = cx; ctx.cy = cy; ctx.cz = cz; ctx.gx = grid.x; ctx.gy = grid.y;
  // A cluster's CTAs wait on each other (its barrier, its peers' shared memory): the whole grid is
  // launched cooperatively, so every CTA is resident at once, or the run refuses.
  uint64_t clusters = ctas / csize;
  size_t dyn = 0;
  std::vector<unsigned long long> bars0(clusters, (unsigned long long)(csize * LIFT_WARPS) << 20);
  unsigned long long* dBars; CK(cudaMalloc(&dBars, 8 * clusters));
  ctx.clusterBars = dBars;
  uint8_t* dTmem; CK(cudaMalloc(&dTmem, lift::TMEM_BYTES * ctas));
  ctx.tmemAll = dTmem;
  if (csize > 1) {
    int perSM = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, lifted, LIFT_THREADS, dyn));
    if ((uint64_t)perSM * prop.multiProcessorCount < ctas) {
      fprintf(stderr, "a clustered grid of %llu CTAs does not fit at once (%d per SM)\n", (unsigned long long)ctas, perSM);
      return 1;
    }
  }

  auto run = [&](float* ms) {
    CK(cudaMemset(dShared, 0, sharedBytes * ctas));
    // Each CTA's reserved shared memory as the simulator starts it (`sharedInit<rank>`).
    for (uint64_t c = 0; c < ctas; ++c) {
      uint32_t x = c % grid.x, y = (c / grid.x) % grid.y, z = c / ((uint64_t)grid.x * grid.y);
      uint32_t rank = x % cx + cx * (y % cy + cy * (z % cz));
      auto it = meta.find("sharedInit" + std::to_string(rank));
      if (it == meta.end()) continue;
      for (size_t j = 0; j + 1 < it->second.size(); j += 2) {
        uint32_t w = (uint32_t)it->second[j + 1];
        CK(cudaMemcpy(dShared + c * sharedBytes + it->second[j], &w, 4, cudaMemcpyHostToDevice));
      }
    }
    CK(cudaMemset(dFault, 0, 4));
    CK(cudaMemcpy(dBars, bars0.data(), 8 * clusters, cudaMemcpyHostToDevice));
    CK(cudaMemset(dTmem, 0, lift::TMEM_BYTES * ctas));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    if (csize > 1) {
      void* args[] = {&ctx};
      CK(cudaLaunchCooperativeKernel((void*)lifted, grid, block, args, dyn, 0));
    } else {
      lifted<<<grid, block, dyn>>>(ctx);
    }
    CK(cudaEventRecord(b));
    CK(cudaGetLastError());
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  };

  float first; run(&first);
  uint32_t fault; CK(cudaMemcpy(&fault, dFault, 4, cudaMemcpyDeviceToHost));
  std::vector<uint8_t> out(global.size());
  CK(cudaMemcpy(out.data(), dGlobal, out.size(), cudaMemcpyDeviceToHost));
  if (out1) {
    auto save = [&](const std::string& name, const std::vector<uint8_t>& h) {
      std::ofstream f(dir + "/" + name, std::ios::binary);
      f.write((const char*)h.data(), (std::streamsize)h.size());
      if (!f) { fprintf(stderr, "cannot write %s\n", name.c_str()); exit(1); }
    };
    save("out.bin", out);
    if (hasModule) {
      std::vector<uint8_t> m(module.size());
      if (!m.empty()) CK(cudaMemcpy(m.data(), dModule, m.size(), cudaMemcpyDeviceToHost));
      save("out-module.bin", m);
    }
    printf("fault flags %#x, kernel time %.3f ms\n", fault, first);
    return fault == 0 ? 0 : 4;
  }
  size_t n = std::min(out.size(), final_.size()), bad = 0;
  std::vector<size_t> where;
  for (size_t i = 0; i < n; ++i) if (out[i] != final_[i]) { if (where.size() < 8) where.push_back(i); ++bad; }
  size_t changed = 0;
  for (size_t i = 0; i < n; ++i) if (final_[i] != global[i]) ++changed;
  printf("fault flags %#x\n", fault);
  printf("bytes the simulator's run changed: %zu; bytes differing from the simulator's result: %zu of %zu\n", changed, bad, n);
  for (size_t i : where) printf("  differs at +%#zx (address %#llx): lifted %02x, simulator %02x, before %02x\n",
                                i, (unsigned long long)(globalBase + i), out[i], final_[i], global[i]);

  std::vector<float> t;
  for (int k = 0; k < reps; ++k) { float ms; run(&ms); t.push_back(ms); }
  std::sort(t.begin(), t.end());
  printf("kernel time: first %.3f ms, median of %d %.3f ms, min %.3f ms\n", first, reps, t[t.size() / 2], t[0]);
  printf("%s\n", bad == 0 && fault == 0 ? "BIT-EXACT" : "MISMATCH");
  return bad == 0 && fault == 0 ? 0 : 3;
}
