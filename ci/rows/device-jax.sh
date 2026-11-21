# JAX on the emulated CUDA device: jax.devices() shows the part, and a float32 and a bfloat16
# matmul, a softmax and a one-layer transformer's loss and gradients run through XLA, NVIDIA's math
# libraries and the driver (tools/cudadev/jax_check.py); their outputs are kept as artifacts, to
# bit-compare with the same programs on the real GPU. And the launch-result cache: hit,
# bit-identical, and missed when a model parameter changes.
CUDA="$HOME/cuda" sh tools/cudadev/build.sh > /dev/null
dir=$(pwd)/.lake/build/cudadev
art=${ARMADA_ARTIFACTS:-$dir}
python3 -m venv "$HOME/jax12env"
"$HOME/jax12env/bin/pip" install -q "jax[cuda12]==0.12.0.1"
for gpu in ${JAX_GPUS:-h100 a100-80}; do
  LD_LIBRARY_PATH="$dir" NVGPU_LEAN_GPU=$gpu NVGPU_LEAN_CUDA="$HOME/cuda" NVGPU_LEAN_CACHE="$dir/cache" \
    NVGPU_LEAN_CUOBJDUMP_OLD=$(sh tools/cudadev/cuobjdump.sh 12.8 "$HOME/cuobjdumps") NVGPU_LEAN_CUOBJDUMP_NEW=$(sh tools/cudadev/cuobjdump.sh 13.5 "$HOME/cuobjdumps") \
    NVGPU_LEAN_VERBOSE=1 NVGPU_LEAN_TRACE=1 "$HOME/jax12env/bin/python" tools/cudadev/jax_check.py run "$art/jax-$gpu.npz" \
    2> "$art/jax-$gpu.err" || { grep -v "is not provided\|^nvgpu-lean: cd \|^nvgpu-lean: rm " "$art/jax-$gpu.err" | tail -80; exit 1; }
done
# The launch-result cache (Device/Cache.lean), timed: a second run of the same cases simulates no
# launch and bit-matches the first; one model parameter changed (the clock) misses every entry.
c="$art/launch-cache"; rm -rf "$c"
lc() {
  tag=$1; shift
  env "$@" LD_LIBRARY_PATH="$dir" NVGPU_LEAN_GPU=h100 NVGPU_LEAN_TIMED=1 NVGPU_LEAN_VERBOSE=1 \
    NVGPU_LEAN_LAUNCH_CACHE="$c" NVGPU_LEAN_CUDA="$HOME/cuda" NVGPU_LEAN_CACHE="$dir/cache" \
    NVGPU_LEAN_CUOBJDUMP_OLD=$(sh tools/cudadev/cuobjdump.sh 12.8 "$HOME/cuobjdumps") NVGPU_LEAN_CUOBJDUMP_NEW=$(sh tools/cudadev/cuobjdump.sh 13.5 "$HOME/cuobjdumps") \
    "$HOME/jax12env/bin/python" tools/cudadev/jax_check.py run "$art/lc-$tag.npz" matmul_f32 softmax 2> "$art/lc-$tag.err" \
    || { tail -40 "$art/lc-$tag.err"; exit 1; }
  echo "launch cache, $tag: $(grep -c "the engine's run" "$art/lc-$tag.err") simulated, $(grep -c "launch cache hit" "$art/lc-$tag.err") hit"
}
lc first; lc again; lc clock NVGPU_LEAN_CLOCK=1000
[ "$(grep -c "the engine's run" "$art/lc-first.err")" -gt 0 ] || exit 1
[ "$(grep -c "the engine's run" "$art/lc-again.err")" -eq 0 ] || exit 1
[ "$(grep -c "launch cache hit" "$art/lc-again.err")" -gt 0 ] || exit 1
[ "$(grep -c "launch cache hit" "$art/lc-clock.err")" -eq 0 ] || exit 1
"$HOME/jax12env/bin/python" tools/cudadev/jax_check.py compare "$art/lc-first.npz" "$art/lc-again.npz"
