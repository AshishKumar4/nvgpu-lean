#!/bin/sh
# docs/sass-coverage.md: NVIDIA's per-family opcode tables against the compiled listings of each
# target (the fixtures, tests/diff/*.ptx and cuda-samples, compiled here) and the semantics.
# Usage: tools/sass/coverage.sh (from the repository root; needs ptxas and cuobjdump).
set -eu
tmp=$(mktemp -d)
sh tools/ptx/corpus.sh "$tmp/c" samples > /dev/null 2>&1
args=""
for arch in sm_80 sm_90a sm_120a; do
  mkdir -p "$tmp/$arch"
  for f in tests/diff/*.ptx "$tmp"/c/samples/*.sm_90a.ptx; do
    n=$(basename "$f" .ptx)
    sed "s/^\.target sm_90a/.target $arch/" "$f" > "$tmp/x.ptx"
    ptxas -arch=$arch -m64 "$tmp/x.ptx" -o "$tmp/x.cubin" > /dev/null 2>&1 || continue
    cuobjdump -sass "$tmp/x.cubin" > "$tmp/$arch/$n.sass"
    cuobjdump -elf "$tmp/x.cubin" > "$tmp/$arch/$n.elf"
  done
  args="$args $arch=$tmp/$arch"
done
lake build sass-coverage
.lake/build/bin/sass-coverage --doc $args > docs/sass-coverage.md
.lake/build/bin/sass-coverage $args
rm -rf "$tmp"
