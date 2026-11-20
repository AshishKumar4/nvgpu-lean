#!/bin/sh
# Regenerate the reader's fixtures with the pinned CUDA tools (tools/cuda/install.sh): for each
# PTX or CUDA source here, the cubin's `cuobjdump -sass` (NAME.sass), `nvdisasm -hex -c`
# (NAME.nvdisasm) and `cuobjdump -elf` (NAME.elf) listings. `--check` regenerates into a temporary
# directory and fails if any fixture differs.
#
# For the other targets (`sm_80/`, `sm_120a/`, `sm_100a/`), the `cuobjdump -sass` and `-elf` listings of every
# kernel source the tests use (here, ../sem, ../diff, and ../large: larger kernels that set a fact), with each PTX file's `.target` set to that
# target; SKIPPED lists the sources that target cannot compile (instructions it lacks).
set -eu
cd "$(dirname "$0")"
out=.
srcs="*.ptx *.cu ../sem/*.ptx ../diff/*.ptx ../large/*.ptx"
if [ "${1:-}" = "--check" ]; then out=$(mktemp -d); fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for arch in sm_80 sm_120a sm_100a; do
  mkdir -p "$out/$arch"
  : > "$out/$arch/SKIPPED"
  for src in $srcs; do
    base=$(basename "$src")
    name=${base%.*}
    case "$src" in
      ../diff/*) name="diff-$name" ;;
      ../sem/*) name="sem-$name" ;;
      ../large/*) name="large-$name" ;;
    esac
    rm -f "$tmp/k.cubin"
    case "$src" in
      # A source may name ptxas flags in a `// ptxas-flags:` line (`-c`: a relocatable module).
      *.ptx) sed -E "s/^\.target sm_90a?\b/.target $arch/" "$src" > "$tmp/k.ptx"
             ptxas -arch=$arch -m64 $(sed -n 's|^// ptxas-flags: ||p' "$src" | head -1) "$tmp/k.ptx" -o "$tmp/k.cubin" \
               > /dev/null 2>&1 || true ;;
      *.cu) nvcc -arch=$arch -cubin -o "$tmp/k.cubin" "$src" \
              > /dev/null 2>&1 || true ;;
    esac
    if [ ! -s "$tmp/k.cubin" ]; then echo "$name" >> "$out/$arch/SKIPPED"; continue; fi
    cuobjdump -sass "$tmp/k.cubin" > "$out/$arch/$name.sass"
    cuobjdump -elf "$tmp/k.cubin" > "$out/$arch/$name.elf"
  done
done
for src in *.ptx *.cu; do
  name=${src%.*}
  case "$src" in
    *.ptx) ptxas -arch=sm_90a -m64 $(sed -n 's|^// ptxas-flags: ||p' "$src" | head -1) "$src" -o "$tmp/$name.cubin" ;;
    *.cu) nvcc -arch=sm_90a -cubin -o "$tmp/$name.cubin" "$src" ;;
  esac
  cuobjdump -sass "$tmp/$name.cubin" > "$out/$name.sass"
  nvdisasm -hex -c "$tmp/$name.cubin" > "$out/$name.nvdisasm"
  cuobjdump -elf "$tmp/$name.cubin" > "$out/$name.elf"
done
if [ "$out" != . ]; then
  status=0
  for f in "$out"/* "$out"/sm_*/*; do
    [ -f "$f" ] || continue
    rel=${f#"$out"/}
    cmp -s "$f" "$rel" || { echo "fixture differs: $rel" >&2; status=1; }
  done
  rm -rf "$out"
  exit $status
fi
