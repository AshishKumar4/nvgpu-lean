#!/bin/sh
# ROCetta's PTX semantics in C (University of Rochester, LGPL-3.0-or-later; the 2026-02 beta
# release) as a differential oracle: download it at a pinned hash, run every scalar function on
# test vectors (harness.py), and write the results to OUT. Nothing of it is vendored here.
# Usage: tools/ptx/rocetta/run.sh OUT. Needs curl, gcc and python3.
set -eu
out=$(realpath -m "$1")
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
curl -fsSL -o c.tar.bz2 https://cs.rochester.edu/~sree/rocetta-semantics/ptx-semantics-c.tar.bz2
echo "5ce2f530254da011c9123af3403bf295cdfb5223099054d0dcb69daf104fff99  c.tar.bz2" | sha256sum -c --quiet
tar xjf c.tar.bz2
# The C runtime the release leaves out, from ptx-xir at a pinned commit.
xir=https://raw.githubusercontent.com/pyxis-roc/ptx-xir/b1629eb84a525dd156f86743f67d46cf459cfe4f/c
curl -fsSL -o c/128types.h "$xir/128types.h"
curl -fsSL -o c/ptxc_utils.h "$xir/ptxc_utils.h"
python3 "$here/harness.py" c/ptxc.c > c/harness.c
# Directed rounding through fesetround: no contraction, no constant folding across modes.
gcc -O0 -frounding-math -ffp-contract=off -w -I c -o harness c/harness.c c/ptxc.c -lm
./harness > "$out"
wc -l < "$out"
