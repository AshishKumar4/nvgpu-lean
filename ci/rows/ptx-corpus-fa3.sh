# PTX that nvcc emits for FlashAttention-3 forward and backward instantiations reads and
# type-checks.
c=/tmp/ptx-corpus
sh tools/ptx/corpus.sh "$c" fa3
cat "$c"/fa3.log
lake build ptx
lake exe ptx check "$c/fa3"
