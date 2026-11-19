# The PTX semantics on nvcc-compiled kernels, checked against references and ROCetta's C semantics.
lake build ptx ptx-tests ptx-oracle && lake exe ptx check tests/ptx && lake exe ptx-tests &&
  sh tools/ptx/rocetta/run.sh /tmp/rocetta.txt && lake exe ptx-oracle /tmp/rocetta.txt
