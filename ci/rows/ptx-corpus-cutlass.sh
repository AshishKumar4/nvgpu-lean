# PTX that nvcc emits for eight CUTLASS v4.2.1 Hopper and Blackwell GEMM examples reads and
# type-checks.
c=/tmp/ptx-corpus
sh tools/ptx/corpus.sh "$c" cutlass
cat "$c"/cutlass.log
lake build ptx
lake exe ptx check "$c/cutlass"
