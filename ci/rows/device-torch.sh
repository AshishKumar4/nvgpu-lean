# PyTorch on the emulated CUDA device: torch_check.py's cases on PyTorch's own kernels (a softmax,
# reductions, elementwise maths, a layer norm) on an H100 and an A100; their outputs are kept as
# artifacts, to bit-compare with the same programs on the GPUs. Its matmuls are not run: NVIDIA's
# math libraries ask the runtime for a table entry the device does not provide (README).
CUDA="$HOME/cuda" sh tools/cudadev/build.sh > /dev/null
dir=$(pwd)/.lake/build/cudadev
art=${ARMADA_ARTIFACTS:-$dir}
python3 -m venv "$HOME/torchenv"
"$HOME/torchenv/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple "torch==2.9.1" numpy
for gpu in ${TORCH_GPUS:-h100 a100-80}; do
  LD_LIBRARY_PATH="$dir" NVGPU_LEAN_GPU=$gpu NVGPU_LEAN_CUDA="$HOME/cuda" NVGPU_LEAN_CACHE="$dir/cache" \
    NVGPU_LEAN_CUOBJDUMP_OLD=$(sh tools/cudadev/cuobjdump.sh 12.8 "$HOME/cuobjdumps") NVGPU_LEAN_CUOBJDUMP_NEW=$(sh tools/cudadev/cuobjdump.sh 13.5 "$HOME/cuobjdumps") \
    NVGPU_LEAN_VERBOSE=1 NVGPU_LEAN_TRACE=1 "$HOME/torchenv/bin/python" tools/cudadev/torch_check.py run "$art/torch-$gpu.npz" softmax reductions layer_norm \
    2> "$art/torch-$gpu.err" || { grep -v "is not provided\|^nvgpu-lean: cd \|^nvgpu-lean: rm " "$art/torch-$gpu.err" | tail -80; exit 1; }
done
