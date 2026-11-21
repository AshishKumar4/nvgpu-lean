# The simulator's speed on whole kernels, per commit (docs/speed.md): wall time and peak memory, with
# the simulated cycles and warp instructions, for a 64 K-element axpy, the TMA + wgmma GEMM (H100,
# timed and functional) and the tcgen05 GEMM (B200), on the default workers.
set -eu
lake build axpy sim
/usr/bin/time -f "axpy 65536: wall %e s, peak %M KB" .lake/build/bin/axpy 65536 2>&1 | grep -E "simulated|wall"
w=$(mktemp -d)
timed() { # tag args...
  tag=$1; shift
  /usr/bin/time -f "$tag: wall %e s, peak %M KB" -o "$w/$tag.time" .lake/build/bin/sim "$@" > "$w/$tag.out" 2>&1 ||
    { cat "$w/$tag.out"; exit 1; }
  grep -E "instructions|cycles" "$w/$tag.out" | head -2
  cat "$w/$tag.time"
}
timed gemm-h100 examples/specs/gemm_tma_wgmma.json --gpu h100
timed gemm-h100-functional examples/specs/gemm_tma_wgmma.json --gpu h100 --functional
timed tcgen05-b200 examples/specs/tcgen05_gemm.json --gpu b200
