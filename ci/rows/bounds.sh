# Gap reports for every part (NV.Arch.parts): axpy compiled by the pinned ptxas for the part's
# target and, where the part has TMA and warpgroup MMA, the GEMM tile kernel from its launch spec,
# each run on the part against its bound with the run checked against the rate invariants; the
# production-size GEMM and attention bounds; the decode-layer floors on the H200.
lake build gap || exit 1
out="${CI_ARTIFACTS:-/tmp}/gap.md"
lake exe gap 65536 > "$out"; s=$?
cat "$out"
exit $s
