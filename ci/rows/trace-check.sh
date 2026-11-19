# The example kernels' warps, as the engine ran them, replay against the program semantics
# (`lake exe trace-check`, `NV.Formal.Trace.check`): each such warp ran a run of `Outcomes`.
lake build trace-check
for gpu in h200 h100 a100-40; do
  for f in examples/specs/axpy.json examples/specs/reduce.json; do lake exe trace-check "$f" --gpu "$gpu"; done
done
for gpu in h200 h100; do
  lake exe trace-check examples/specs/cluster_dsmem.json --gpu "$gpu"
  lake exe trace-check examples/specs/gemm_tma_wgmma.json --gpu "$gpu"
done
