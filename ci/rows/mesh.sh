# JAX on meshes of emulated devices (tools/cudadev/mesh_check.py): two H100s and eight B200s are
# each seen, run jitted programs and pass arrays between them; pmap and shard_map match one device
# bit for bit, and collectives (psum, all_gather, psum_scatter, ppermute, all_to_all) run on the
# emulated NCCL. NVML (nvidia-smi -L) lists every device.
CUDA="$HOME/cuda" sh tools/cudadev/build.sh > /dev/null
dir=$(pwd)/.lake/build/cudadev
python3 -m venv "$HOME/jaxenv"
"$HOME/jaxenv/bin/pip" install -q "jax[cuda13]==0.12.0.1"
for gpu in h100x2 b200x8; do
  echo "== $gpu"
  LD_LIBRARY_PATH="$dir" NVGPU_LEAN_GPU=$gpu "$dir/nvidia-smi" -L
  LD_LIBRARY_PATH="$dir" NVGPU_LEAN_GPU=$gpu NVGPU_LEAN_CUDA="$HOME/cuda" NVGPU_LEAN_CACHE="$dir/cache" \
    NVGPU_LEAN_CUOBJDUMP_OLD=$(sh tools/cudadev/cuobjdump.sh 12.8 "$HOME/cuobjdumps") NVGPU_LEAN_CUOBJDUMP_NEW=$(sh tools/cudadev/cuobjdump.sh 13.5 "$HOME/cuobjdumps") \
    NVGPU_LEAN_VERBOSE=1 "$HOME/jaxenv/bin/python" tools/cudadev/mesh_check.py 2> err.txt ||
    { grep -v "is not provided\|^nvgpu-lean: cd \|^nvgpu-lean: rm " err.txt | tail -60; exit 1; }
  # The collectives ran on the emulated NCCL (each call logged), not NVIDIA's, and took time.
  echo "emulated NCCL calls: $(grep -c '^nvgpu-lean NCCL: rank' err.txt)"
  grep -q '^nvgpu-lean NCCL: rank' err.txt
  grep '^nvgpu-lean NCCL: AllGather of\|^nvgpu-lean: device .* simulated$' err.txt | head -12
done
# The relocatable bundle (tools/cudadev/bundle.sh) runs the same check on two H100s with only its
# env.sh set (env -i): nothing of this checkout but mesh_check.py itself.
CUDA="$HOME/cuda" sh tools/cudadev/bundle.sh /tmp/nvgpu-lean-bundle.tar.gz > /dev/null
cp /tmp/nvgpu-lean-bundle.tar.gz "${ARMADA_ARTIFACTS:-/tmp}/" 2>/dev/null || true
x=$(mktemp -d); tar xzf /tmp/nvgpu-lean-bundle.tar.gz -C "$x"; cp tools/cudadev/mesh_check.py "$x/"
echo "== bundle on h100x2: $(du -sh /tmp/nvgpu-lean-bundle.tar.gz | cut -f1)"
(cd "$x" && env -i HOME="$HOME" PATH=/usr/bin:/bin sh -c ". nvgpu-lean/env.sh && nvidia-smi -L && '$HOME/jaxenv/bin/python' mesh_check.py") 2> err.txt ||
  { tail -40 err.txt; exit 1; }
