# ptxas-compiled kernels with TMA, bulk copies, cp.async, mbarrier rings and wgmma, run end to end
# in the engine and checked exactly (tests/kernels/async_copies.cu, gemm_tma_wgmma.cu).
# A panic in the engine (an index past an array) is a defect, not a default value: abort on it.
export LEAN_ABORT_ON_PANIC=1
lake build async-copies async-gemm && lake exe async-copies && lake exe async-gemm
