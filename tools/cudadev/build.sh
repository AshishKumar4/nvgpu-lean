#!/bin/sh
# Build the emulated device's libcuda.so.1 (tools/cudadev/README.md): lake builds the Lean side's
# shared libraries, gen.py writes parts.h, procs.h and stubs.c, and the C compiler links the shim.
# CUDA: the toolkit whose cuda.h the shim is compiled against (default /usr/local/cuda).
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
cuda=${CUDA:-/usr/local/cuda}
out=${OUT:-$root/.lake/build/cudadev}
mkdir -p "$out"
cd "$root"
lake build NV:shared Device:shared >&2
# nvml.h: the toolkit's, or the NVML development package's.
nvmlh=$(ls "$cuda"/include/nvml.h "$cuda"/targets/*/include/nvml.h /usr/include/nvml.h 2>/dev/null | head -1)
python3 tools/cudadev/gen.py "$root" "$cuda/include/cuda.h" tools/cudadev/cuda.c "$out" \
  ${nvmlh:+"$nvmlh" tools/cudadev/nvml.c} >&2
prefix=$(lean --print-prefix)
lib=$root/.lake/build/lib
nvlib=$(ls "$lib"/lib*NV.so | head -1)
devlib=$(ls "$lib"/lib*Device.so | head -1)
# The Lean libraries are linked by name, not path, so the library loads from wherever they are
# found (its rpath, or LD_LIBRARY_PATH: a copy on another machine).
# The Lean module initializer's symbol, as this toolchain names it.
init=$(nm -D --defined-only "$devlib" | awk '/ initialize_.*Device_Api$/ {print $3}' | head -1)
cc -shared -fPIC -O2 -Wall -Wno-unused-function -Wno-deprecated-declarations -DLEAN_INIT="$init" \
  -o "$out/libcuda.so.1" tools/cudadev/cuda.c tools/cudadev/legacy.c "$out/stubs.c" \
  -I"$out" -I"$cuda/include" -I"$prefix/include" \
  -L"$lib" -l"$(basename "$devlib" .so | sed 's/^lib//')" -l"$(basename "$nvlib" .so | sed 's/^lib//')" \
  -L"$prefix/lib/lean" -lleanshared \
  -Wl,-rpath,"$lib" -Wl,-rpath,"$prefix/lib/lean" -lpthread -ldl
ln -sf libcuda.so.1 "$out/libcuda.so"
# The NCCL a mesh's collectives run on (nccl.c), over this libcuda.
cc -shared -fPIC -O2 -Wall -Wno-unused-parameter -o "$out/libnccl.so.2" tools/cudadev/nccl.c \
  -I"$cuda/include" -L"$out" -lcuda -Wl,-rpath,"$out" -lm -lpthread
if [ -n "$nvmlh" ]; then
  cc -shared -fPIC -O2 -Wall -Wno-unused-parameter -Wno-unused-function -o "$out/libnvidia-ml.so.1" \
    tools/cudadev/nvml.c -I"$out" -I"$(dirname "$nvmlh")" -I"$cuda/include"
  ln -sf libnvidia-ml.so.1 "$out/libnvidia-ml.so"
  cc -O2 -Wall -Wno-format-truncation -o "$out/nvidia-smi" tools/cudadev/smi.c -I"$(dirname "$nvmlh")" \
    -L"$out" -lnvidia-ml -Wl,-rpath,"$out"
else
  echo "no nvml.h: libnvidia-ml.so.1 not built" >&2
fi
echo "$out/libcuda.so.1"
