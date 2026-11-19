# The SASS semantics on `sm_100a` (B200): the control words of its fixtures pair up (scoreboards),
# every fixture instruction has semantics and stays within its footprint, and the semantics and
# differential tests pass on kernels compiled for it, tensor memory and tcgen05 among them (those
# it cannot compile, wgmma, are skipped).
export NV_ARCH=sm_100a
lake build sass-ctrl sass-sem sass-diff sass-footprint
lake exe sass-ctrl tests/sass/sm_100a
lake exe sass-sem
lake exe sass-diff
lake exe sass-footprint
