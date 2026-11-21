#!/bin/sh
# A relocatable copy of the emulated device for machines without this repository: the driver,
# NVML, NCCL and nvidia-smi (build.sh), the Lean libraries they load, the toolkit's ptxas, cuobjdump
# and nvdisasm (which every cuobjdump runs, from PATH) and the two other cuobjdumps the driver lists cubins with (cuobjdump.sh), jax_mesh.py,
# and env.sh, which points a shell at them:
#   sh tools/cudadev/bundle.sh OUT.tar.gz          # CUDA: the toolkit (default /usr/local/cuda)
#   tar xzf OUT.tar.gz && . nvgpu-lean/env.sh       # NVGPU_LEAN_GPU defaults to h100x2
#   python nvgpu-lean/python/jax_mesh.py PROGRAM.py
# The libraries need the glibc of the machine that built them, or a newer one.
set -e
root=$(cd "$(dirname "$0")/../.." && pwd)
cuda=${CUDA:-/usr/local/cuda}
out=$1
[ -n "$out" ] || { echo "usage: sh tools/cudadev/bundle.sh OUT.tar.gz" >&2; exit 2; }
lib=$(CUDA="$cuda" sh "$root/tools/cudadev/build.sh" | tail -1)
[ -f "$lib" ] || { echo "bundle.sh: build.sh built no libcuda.so.1 (CUDA=$cuda)" >&2; exit 1; }
dir=$(dirname "$lib")
stage=$(mktemp -d)
b=$stage/nvgpu-lean
mkdir -p "$b/lib" "$b/bin" "$b/cuda/bin" "$b/python"
cp -L "$dir/libcuda.so.1" "$dir/libnccl.so.2" "$b/lib/"
ln -s libcuda.so.1 "$b/lib/libcuda.so"
if [ -f "$dir/libnvidia-ml.so.1" ]; then
  cp -L "$dir/libnvidia-ml.so.1" "$b/lib/" && ln -s libnvidia-ml.so.1 "$b/lib/libnvidia-ml.so"
  cp "$dir/nvidia-smi" "$b/bin/"
fi
# The Lean libraries the driver links by name (NV, Device, the runtime), and whatever of the Lean
# toolchain's own those load.
cp -L "$root"/.lake/build/lib/lib*NV.so "$root"/.lake/build/lib/lib*Device.so "$b/lib/"
prefix=$(lean --print-prefix)
for f in $(ldd "$b/lib/libcuda.so.1" | awk -v p="$prefix" 'index($3, p) == 1 {print $3}'); do cp -L "$f" "$b/lib/"; done
cp -L "$cuda/bin/ptxas" "$cuda/bin/cuobjdump" "$cuda/bin/nvdisasm" "$b/cuda/bin/"
old=$(sh "$root/tools/cudadev/cuobjdump.sh" 12.8 "$stage/fetch"); new=$(sh "$root/tools/cudadev/cuobjdump.sh" 13.5 "$stage/fetch")
mkdir -p "$b/cuobjdump-12.8" "$b/cuobjdump-13.5"
cp "$old" "$b/cuobjdump-12.8/"; cp "$new" "$b/cuobjdump-13.5/"
cp "$root/tools/cudadev/jax_mesh.py" "$root/tools/cudadev/jax_check.py" "$b/python/"
cat > "$b/env.sh" <<'ENV'
# The emulated device from this directory: `. nvgpu-lean/env.sh` (bash, sh, zsh).
here=$(cd "$(dirname "${BASH_SOURCE:-${ZSH_ARGZERO:-$0}}")" 2>/dev/null && pwd)
[ -f "$here/lib/libcuda.so.1" ] || here=$(pwd)/nvgpu-lean
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$here/bin:$here/cuda/bin:$PATH"
export PYTHONPATH="$here/python${PYTHONPATH:+:$PYTHONPATH}"
export NVGPU_LEAN_GPU="${NVGPU_LEAN_GPU:-h100x2}"
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1  # the device has no profiler (CUPTI), whose version JAX checks
export NVGPU_LEAN_CUDA="$here/cuda"
export NVGPU_LEAN_CUOBJDUMP_OLD="$here/cuobjdump-12.8/cuobjdump"
export NVGPU_LEAN_CUOBJDUMP_NEW="$here/cuobjdump-13.5/cuobjdump"
export NVGPU_LEAN_CACHE="${NVGPU_LEAN_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/nvgpu-lean}"
ENV
git -C "$root" rev-parse HEAD > "$b/VERSION"
rm -rf "$stage/fetch"
tar czf "$out" -C "$stage" nvgpu-lean
rm -rf "$stage"
echo "$out"
