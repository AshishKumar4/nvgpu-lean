# The accelerated mode's lifter (tools/lift) against the simulator: every example spec's launches
# exported on the H100, A100 and B200 parts, each lifted from the simulator's own parse and the
# lifted kernel compiled by the pinned nvcc for sm_80 and sm_90a (tools/lift/check.sh --objects).
# The exports, lifted, are kept as an artifact (lift-exports.tgz): on a GPU, `check.sh` over them
# runs each launch there and compares it with the simulator's result byte for byte.
set -eu
lake build sim
out=$(mktemp -d)
# The examples each part runs (the examples row's lists), and the special-function sweep.
export_on() { # gpu spec...
  gpu=$1; shift
  for n in "$@"; do lake exe sim "examples/specs/$n.json" --gpu "$gpu" --functional --export "$out/$gpu/$n"; done
}
export_on h100 axpy reduce reduce_grid gemm_tma_wgmma cluster_dsmem tma_multicast special
export_on a100-40 axpy reduce reduce_grid special
export_on b200 axpy reduce tcgen05_gemm tcgen05_pair special
sh tools/lift/check.sh --objects --arch sm_80,sm_90a "$out"
if [ -n "${ARMADA_ARTIFACTS:-}" ]; then
  find "$out" -name '*.o' -delete
  tar czf "$ARMADA_ARTIFACTS/lift-exports.tgz" -C "$out" .
fi
