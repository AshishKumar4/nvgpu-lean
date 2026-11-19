#!/bin/sh
# Builds the PTX corpus the front end is tested on, CPU-only, from pinned sources:
#   ours     tests/ptx/kernels/*.cu
#   samples  NVIDIA cuda-samples v13.0, every sample that compiles stand-alone
#   cutlass  CUTLASS v4.2.1, eight Hopper and Blackwell GEMM examples
#   fa3      FlashAttention-3 (flash-attention's hopper/): forward and backward instantiations
#   triton   Triton 3.5.0 kernels of tools/ptx/triton_kernels.py
# Usage: tools/ptx/corpus.sh DEST [STAGE...] (default: every stage). PTX lands in DEST/<stage>/;
# DEST/<stage>.log lists what failed to compile. Needs nvcc (tools/cuda/install.sh), g++, git and
# python3 with venv. Sources are checked out at pinned commits; a moved tag fails the build.
set -eu
dest=$(realpath -m "$1"); shift
stages=${*:-ours samples cutlass fa3 triton}
root=$(cd "$(dirname "$0")/../.." && pwd)
src="$dest/src"
mkdir -p "$dest" "$src"

# checkout NAME URL REF SHA: a shallow clone of REF, which must be commit SHA.
checkout() {
  if [ ! -d "$src/$1" ]; then
    git clone -q --depth 1 --branch "$3" "$2" "$src/$1" 2>/dev/null ||
      { git init -q "$src/$1" && git -C "$src/$1" fetch -q --depth 1 "$2" "$3" &&
        git -C "$src/$1" checkout -q FETCH_HEAD; }
  fi
  got=$(git -C "$src/$1" rev-parse HEAD)
  [ "$got" = "$4" ] || { echo "$1: $3 is $got, expected $4" >&2; exit 1; }
}

# Device-side headers some sources include beyond tools/cuda: cuRAND (curand_kernel.h) and the
# profiler API, CUDA 13.0's wheels, headers only, into $src/include.
headers() {
  [ -d "$src/include" ] && return
  pypi=https://files.pythonhosted.org/packages
  for w in "a5/9f/be0a41ca4a4917abf5cb9ae0daff1a6060cc5de950aec0396de9f3b52bc5/nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl 1aee33a5da6e1db083fe2b90082def8915f30f3248d5896bcec36a579d941bfc" \
           "98/f8/c59269f2f0f083d1401b65b9e4930f41d1da7aeeec94bd65a4c98c1c56f8/nvidia_cuda_profiler_api-13.0.85-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 73d2671deac833db6686d6fbde4753fa3a3a497d87540fc246f081f375243f66"; do
    set -- $w
    curl -fsSL -o "$src/w.whl" "$pypi/$1"
    echo "$2  $src/w.whl" | sha256sum -c --quiet
    python3 -m zipfile -e "$src/w.whl" "$src/w"
  done
  mkdir -p "$src/include"
  cp -R "$src"/w/nvidia/cu13/include/. "$src/include/"
  rm -rf "$src/w" "$src/w.whl"
}

# ptx ARCHS OUTDIR LOG FILE FLAGS...: FILE to OUTDIR/<flat name>.<arch>.ptx for each arch in
# ARCHS; failures are logged with nvcc's first error lines, not fatal (a sample that needs an
# external library is skipped).
ptx() {
  as=$1; out=$2; log=$3; f=$4; shift 4
  name=$(echo "$f" | sed "s|^$src/||; s|^$root/||; s|/|__|g; s|\.cu$||")
  for a in $as; do
    err="$out/$name.$a.err"
    nvcc -ptx -arch="$a" -I "$src/include" "$@" -o "$out/$name.$a.ptx" "$f" > "$err" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      { echo "FAIL $a $f (exit $rc)"; grep -m 3 error "$err" | sed 's/^/  /'; } >> "$log" || true
      rm -f "$out/$name.$a.ptx"
    fi
    rm -f "$err"
  done
}

# spawn CMD...: run CMD in the background, at most $jobs at once; `wait` collects the rest. Each
# stage sets $jobs to bound memory use.
jobs=$(nproc)
running=0
spawn() {
  "$@" &
  running=$((running + 1))
  if [ "$running" -ge "$jobs" ]; then wait; running=0; fi
}

stage_ours() {
  mkdir -p "$dest/ours"; : > "$dest/ours.log"
  for f in "$root"/tests/ptx/kernels/*.cu; do
    # nvcc does not check inline PTX against the target: a kernel whose instructions a target
    # lacks (mma.cu's stmatrix before sm_90) is left out of that target's PTX, as ptxas would
    # refuse it.
    as=""
    for a in sm_80 sm_90a sm_100a sm_120a; do
      nvcc -cubin -arch="$a" -O3 -o /dev/null "$f" > /dev/null 2>&1 && as="$as $a"
    done
    ptx "$as" "$dest/ours" "$dest/ours.log" "$f" -O3
  done
}

stage_samples() {
  checkout cuda-samples https://github.com/NVIDIA/cuda-samples.git v13.0 \
    3f1c50965017932fc81e6d94a3fc9e04c105b312
  mkdir -p "$dest/samples"; : > "$dest/samples.log"
  jobs=$(nproc)
  for f in $(find "$src/cuda-samples/Samples" -name '*.cu' | sort); do
    spawn ptx sm_90a "$dest/samples" "$dest/samples.log" "$f" -std=c++17 \
      -I "$src/cuda-samples/Common" -I "$(dirname "$f")" -w
  done
  wait
}

stage_cutlass() {
  checkout cutlass https://github.com/NVIDIA/cutlass.git v4.2.1 \
    f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
  mkdir -p "$dest/cutlass"; : > "$dest/cutlass.log"
  jobs=2
  ex="$src/cutlass/examples"
  for f in "$ex"/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu \
           "$ex"/49_hopper_gemm_with_collective_builder/49_collective_builder.cu \
           "$ex"/54_hopper_fp8_warp_specialized_gemm/54_hopper_fp8_warp_specialized_gemm.cu \
           "$ex"/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu \
           "$ex"/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu \
           "$ex"/70_blackwell_gemm/70_blackwell_fp16_gemm.cu \
           "$ex"/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu \
           "$ex"/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu; do
    a=$(case "$f" in *blackwell*) echo sm_100a;; *) echo sm_90a;; esac)
    spawn ptx "$a" "$dest/cutlass" "$dest/cutlass.log" "$f" -std=c++17 --expt-relaxed-constexpr \
      -I "$src/cutlass/include" -I "$src/cutlass/tools/util/include" -I "$ex/common" \
      -I "$(dirname "$f")" -w
  done
  wait
}

stage_fa3() {
  checkout flash-attention https://github.com/Dao-AILab/flash-attention.git \
    2ed4a1f0fc0e36a6ca7b861bfbf86ebb82c4c592 2ed4a1f0fc0e36a6ca7b861bfbf86ebb82c4c592
  sub=$(git -C "$src/flash-attention" ls-tree HEAD csrc/cutlass | awk '{print $3}')
  checkout fa-cutlass https://github.com/NVIDIA/cutlass.git "$sub" "$sub"
  mkdir -p "$dest/fa3"; : > "$dest/fa3.log"
  jobs=1
  h="$src/flash-attention/hopper"
  for k in fwd_hdim128_bf16 fwd_hdim128_e4m3 bwd_hdim128_bf16; do
    f="$h/instantiations/flash_${k}_sm90.cu"
    [ -f "$f" ] || { echo "FAIL missing $f" >> "$dest/fa3.log"; continue; }
    spawn ptx sm_90a "$dest/fa3" "$dest/fa3.log" "$f" -std=c++17 -O3 --use_fast_math \
      --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
      -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_ENABLE_GDC_FOR_SM90 \
      -I "$h" -I "$src/fa-cutlass/include" -w
  done
  wait
}

stage_triton() {
  [ -d "$src/venv" ] || python3 -m venv "$src/venv"
  "$src/venv/bin/pip" -q install triton==3.5.0
  mkdir -p "$dest/triton"
  "$src/venv/bin/python" "$root/tools/ptx/triton_kernels.py" "$dest/triton" > "$dest/triton.log" 2>&1
}

headers
for s in $stages; do "stage_$s"; done
for s in $stages; do
  echo "$s: $(ls "$dest/$s" | wc -l) PTX files, $(grep -c FAIL "$dest/$s.log" || true) failures"
done
