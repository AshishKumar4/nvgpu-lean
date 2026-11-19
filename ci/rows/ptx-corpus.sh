# PTX that nvcc emits for NVIDIA cuda-samples v13.0, our kernels and Triton 3.5.0 kernels reads
# and type-checks, and the committed PTX of tests/ptx/kernels is what nvcc emits.
c=/tmp/ptx-corpus
sh tools/ptx/corpus.sh "$c" ours samples triton
cat "$c"/ours.log "$c"/samples.log "$c"/triton.log
stale=$(for f in tests/ptx/kernels/*.ptx; do
  cmp -s "$f" "$c/ours/tests__ptx__kernels__${f##*/}" || echo "$f"; done)
[ -z "$stale" ] || { echo "not what nvcc emits: $stale"; exit 1; }
lake build ptx
lake exe ptx check "$c/ours" "$c/samples" "$c/triton"
