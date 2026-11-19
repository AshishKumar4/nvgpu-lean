# The accelerated mode's lifter (tools/lift) against the simulator: every example spec's launches
# exported on the H100, A100 and B200 parts, each lifted from the simulator's own parse and the
# lifted kernel compiled by the pinned nvcc for sm_80 and sm_90a (the GPUs that run them; bit-
# exactness against the simulator is checked there, on a GPU: tools/lift/main.cu).
set -eu
lake build sim
out=$(mktemp -d)
# The examples each part runs (the examples row's lists).
export_on() { # gpu spec...
  gpu=$1; shift
  for n in "$@"; do lake exe sim "examples/specs/$n.json" --gpu "$gpu" --functional --export "$out/$gpu/$n"; done
}
export_on h100 axpy reduce reduce_grid gemm_tma_wgmma cluster_dsmem
export_on a100-40 axpy reduce reduce_grid
export_on b200 axpy reduce tcgen05_gemm tcgen05_pair
python3 tools/lift/lift.py --tables "$out/mufu_tables.cuh"
cp tools/lift/rt.cuh tools/lift/main.cu "$out/"
n=0
for js in "$out"/*/*/*/kernel.jsonl; do
  d=$(dirname "$js")
  threads=$(awk '$1 == "block" { print $2 * $3 * $4 }' "$d/meta.txt")
  python3 tools/lift/lift.py --threads "$threads" "$js" > "$d/lifted.cu"
  for arch in sm_80 sm_90a; do
    nvcc -std=c++17 -arch=$arch -ftz=false -fmad=false -I"$out" -I"$d" -x cu -c -o "$d/$arch.o" "$out/main.cu"
  done
  n=$((n + 1))
done
echo "lifted and compiled $n launches"
