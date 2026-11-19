# `lake exe sim` runs the example launch specs in examples/specs (kernels compiled from .cu by the
# pinned nvcc for each part's target) and their outputs are the expected ones: the portable
# examples on every part, the TMA + wgmma GEMM on the parts that have both (the H200's run is in
# `speed`), distributed shared memory in a thread-block cluster on the parts that have clusters.
lake build sim
for gpu in h200 h100 h100-pcie h100-nvl a100-40 a100-80 rtxpro6000 b200 b300; do
  for f in examples/specs/axpy.json examples/specs/reduce.json examples/specs/reduce_grid.json; do lake exe sim "$f" --gpu "$gpu"; done
done
for gpu in h100 h100-pcie h100-nvl; do lake exe sim examples/specs/gemm_tma_wgmma.json --gpu "$gpu"; done
for gpu in h200 h100 h100-pcie h100-nvl rtxpro6000; do
  lake exe sim examples/specs/cluster_dsmem.json --gpu "$gpu"
done
# The functional mode on one worker and on several (global atomics across workers, clusters dealt
# by GPC): the same outputs.
for w in 1 4; do
  for f in reduce reduce_grid cluster_dsmem gemm_tma_wgmma; do lake exe sim examples/specs/$f.json --gpu h100 --functional --workers $w; done
done
# The B200's tensor memory and tcgen05.mma, of one CTA and of a CTA pair: timed and functional.
lake exe sim examples/specs/tcgen05_gemm.json --gpu b200
lake exe sim examples/specs/tcgen05_gemm.json --gpu b200 --functional
lake exe sim examples/specs/tcgen05_pair.json --gpu b200
lake exe sim examples/specs/tcgen05_pair.json --gpu b200 --functional
lake exe sim examples/specs/tcgen05_gemm.json --gpu b300
# --until stops a launch early: the GEMM (4,000 cycles and more whole) stopped at 2000 reports the
# cycles it ran, the stop rounded up to the engine's quantum (within 200).
c=$(lake exe sim examples/specs/gemm_tma_wgmma.json --gpu h100 --until 2000 | sed -n 's/.*start + \([0-9]*\) active.*/\1/p')
echo "gemm_tma_wgmma --until 2000: $c active cycles"; [ "$c" -ge 2000 ] && [ "$c" -le 2200 ]
# --spans reports each CTA once, its end after its start, the last end the launch's last cycle.
lake exe sim examples/specs/gemm_tma_wgmma.json --gpu h100 --spans 2> spans.txt > /dev/null
n=$(grep -c "^cta " spans.txt); g=$(python3 -c "import json; d=json.load(open('examples/specs/gemm_tma_wgmma.json')); import math; print(math.prod(d['grid']))")
echo "gemm_tma_wgmma --spans: $n CTAs of $g"; [ "$n" -eq "$g" ] && awk '/^cta /{ if ($8 <= $6) exit 1 }' spans.txt
