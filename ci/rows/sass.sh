# The SASS reader's fixtures are the pinned tools' output; the reader's tests, the SASS semantics
# tests (ptxas-compiled kernels and writeback timing) and the PTX↔SASS differential test pass, and
# every test kernel instruction stays within its footprint, device printf records render as the
# host prints them, and the device runtime's cudaGraphSetConditional stores the condition.
sh tests/sass/gen.sh --check
lake build sass-tests sass-sem sass-diff sass-footprint sass-printf sass-conditional sass-ctrl
lake exe sass-tests
lake exe sass-sem
lake exe sass-diff
lake exe sass-footprint
lake exe sass-printf
lake exe sass-conditional
lake exe sass-ctrl tests/sass
# A listing that reads an HGMMA's accumulators before its group retires must fail the check.
if lake exe sass-ctrl tests/sass/red > /tmp/sass-ctrl-red.txt; then echo "sass-ctrl accepted tests/sass/red"; exit 1; fi
grep -q "touches its accumulators before the group retires" /tmp/sass-ctrl-red.txt
grep -q "HGMMA with an RZ destination" /tmp/sass-ctrl-red.txt
