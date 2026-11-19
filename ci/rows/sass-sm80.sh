# The SASS semantics on `sm_80`: the control words of its fixtures pair up (scoreboards), every
# fixture instruction has semantics and stays within its footprint, and the semantics and
# differential tests pass on kernels compiled for it (those it cannot compile are skipped).
export NV_ARCH=sm_80
lake build sass-ctrl sass-sem sass-diff sass-footprint
lake exe sass-ctrl tests/sass/sm_80
lake exe sass-sem
lake exe sass-diff
lake exe sass-footprint
