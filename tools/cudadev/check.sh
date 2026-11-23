#!/bin/sh
# Build the emulated device and run its checks on each part: the driver API (test.c), NVML, and
# the runtime API through cudart, which registers each program's fat binary with the driver and
# loads libcuda.so.1 from LD_LIBRARY_PATH (vectorAdd, and runtime.cu: symbols, streams, graphs,
# clusters). Each program checks its own results and exits nonzero on a wrong one.
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
cuda=${CUDA:-/usr/local/cuda}
lib=$(sh "$root/tools/cudadev/build.sh" | tail -1) || exit 1
[ -f "$lib" ] || { echo "build failed"; exit 1; }
dir=$(dirname "$lib")
cc -O2 -o "$dir/test" "$root/tools/cudadev/test.c" -pthread -I"$cuda/include" -L"$dir" -lcuda -Wl,-rpath,"$dir"
nvmlh=$(ls "$cuda"/include/nvml.h "$cuda"/targets/*/include/nvml.h /usr/include/nvml.h 2>/dev/null | head -1)
if [ -f "$dir/libnvidia-ml.so.1" ]; then
  cc -O2 -o "$dir/nvmltest" "$root/tools/cudadev/nvmltest.c" -I"$(dirname "$nvmlh")" -L"$dir" -lnvidia-ml -ldl \
    -Wl,-rpath,"$dir"
fi
# The runtime's library: the toolkit's, or its wheel's (NVIDIA's pip package of libcudart).
rt=$(ls "$cuda"/lib64/libcudart.so* "$cuda"/lib/libcudart.so* "$cuda"/targets/*/lib/libcudart.so* 2>/dev/null | head -1)
if [ -z "$rt" ]; then
  major=$("$cuda/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
  python3 -m pip install -q --target "$dir/rt" "nvidia-cuda-runtime-cu$major" >/dev/null 2>&1 || \
    python3 -m pip install -q --target "$dir/rt" "nvidia-cuda-runtime" >/dev/null 2>&1 || true
  rt=$(find "$dir/rt" -name 'libcudart.so*' 2>/dev/null | head -1)
fi
[ -n "$rt" ] || { echo "no CUDA runtime library"; exit 1; }
# Each runtime program for each part's target: compiled by nvcc, linked by the host compiler
# against the runtime's library.
for t in sm_90a sm_80 sm_100a sm_120a; do
  for p in vectoradd runtime; do
    "$cuda/bin/nvcc" -O2 -arch=$t -c -o "$dir/$p-$t.o" "$root/tools/cudadev/$p.cu"
    c++ -o "$dir/$p-$t" "$dir/$p-$t.o" "$rt" -Wl,-rpath,"$(dirname "$rt")" -ldl -lpthread
  done
done
# One module cache for the checks, fresh: every module is prepared from its image here.
export NVGPU_LEAN_CACHE="$dir/cache"
rm -rf "$NVGPU_LEAN_CACHE"
for gpu in h200 a100-80 b200 rtxpro6000; do
  for mode in "" graph legacy failure; do
    NVGPU_LEAN_GPU=$gpu NVGPU_LEAN_CUDA=$cuda "$dir/test" $mode
  done
  if [ -x "$dir/nvmltest" ]; then
    NVGPU_LEAN_GPU=$gpu "$dir/nvmltest"
    NVGPU_LEAN_GPU=$gpu "$dir/nvidia-smi" --query-gpu=name,memory.total,compute_cap --format=csv,noheader
  fi
  t=$(case $gpu in a100*) echo sm_80 ;; b200) echo sm_100a ;; rtx*) echo sm_120a ;; *) echo sm_90a ;; esac)
  for p in vectoradd runtime; do
    LD_LIBRARY_PATH="$dir" NVGPU_LEAN_GPU=$gpu NVGPU_LEAN_CUDA=$cuda "$dir/$p-$t"
  done
done
# A mesh: a kernel reaching a peer's memory, on two of one part and on two different parts.
for gpu in h100x2 h100,h200; do
  NVGPU_LEAN_GPU=$gpu NVGPU_LEAN_CUDA=$cuda "$dir/test" peer
done
# Family-specific PTX is forward compatible inside its family, never into another family.
# The -arch shorthand also emits generic compute_100 PTX, which validly runs on sm_120.
# Request only compute_100f so this program tests the family restriction itself.
"$cuda/bin/nvcc" -O2 -fatbin --generate-code=arch=compute_100f,code=compute_100f \
  -o "$dir/family.fatbin" "$root/tools/cudadev/vectoradd.cu"
NVGPU_LEAN_GPU=b200 NVGPU_LEAN_CUDA=$cuda "$dir/test" family-ok "$dir/family.fatbin"
NVGPU_LEAN_GPU=rtxpro6000 NVGPU_LEAN_CUDA=$cuda "$dir/test" family-refused "$dir/family.fatbin"
# A program built for another architecture than the part's has no kernel for it: refused as the
# driver refuses it.
if LD_LIBRARY_PATH="$dir" NVGPU_LEAN_GPU=a100-80 NVGPU_LEAN_CUDA=$cuda "$dir/vectoradd-sm_90a" 2>/dev/null; then
  echo "an sm_90a program ran on an sm_80 part"; exit 1
fi
echo "device checks: ok"
