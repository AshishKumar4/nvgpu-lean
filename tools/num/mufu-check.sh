#!/bin/sh
# Check NV.Num.Mufu against Carleton's PTX Math (docs/sources.md `ptxmath`) at the pinned commit:
# build its C model, print both models' outputs for the same 1.4 million inputs, compare.
set -eu
commit=2c7dc926dd85b755446b594d129c134a82f9dde7
dir="${TMPDIR:-/tmp}/ptx-math"
if [ ! -d "$dir" ]; then git clone -q https://github.com/bcarlet/ptx-math "$dir"; fi
git -C "$dir" checkout -q "$commit"
src="$dir/src/ptxm"
cc -std=c11 -O2 -mbmi2 -ffp-contract=off -I"$src" -o "$dir/mufu_ref" tools/num/mufu_ref.c \
  "$src"/*.c "$src"/common/*.c "$src"/reduction/*.c "$src"/tables/*.c -lm
"$dir/mufu_ref" > "$dir/ref.txt"
lake exe num-tests mufu > "$dir/got.txt"
if cmp -s "$dir/ref.txt" "$dir/got.txt"; then
  echo "ok   mufu: $(wc -l < "$dir/ref.txt") outputs equal to PTX Math"
else
  echo "FAIL mufu: outputs differ from PTX Math (reference, then ours):"
  diff "$dir/ref.txt" "$dir/got.txt" | head -40
  exit 1
fi
