# ptxas-compiled kernels with TMA, bulk copies, cp.async, mbarrier rings and wgmma, run end to end
# in the engine and checked exactly (tests/kernels/async_copies.cu, gemm_tma_wgmma.cu).
lake build async-copies async-gemm && lake exe async-copies && lake exe async-gemm
