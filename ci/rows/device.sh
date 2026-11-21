# The emulated CUDA device (tools/cudadev) on each part: the driver API (module variables, streams
# and events, graphs, clusters, tensor maps, occupancy), NVML, and runtime-API programs through
# cudart (fat-binary registration, symbols, graphs); and the engine's functional mode on the
# example specs (outputs checked as in `examples`).
lake build sim Device
CUDA="$HOME/cuda" sh tools/cudadev/check.sh
for gpu in h200 a100-80 rtxpro6000; do
  for f in examples/specs/axpy.json examples/specs/reduce.json; do lake exe sim "$f" --gpu "$gpu" --functional; done
done
lake exe sim examples/specs/gemm_tma_wgmma.json --gpu h200 --functional
lake exe sim examples/specs/cluster_dsmem.json --gpu h200 --functional
