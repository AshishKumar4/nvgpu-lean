#!/bin/sh
# The accelerated mode against the simulator: every launch the simulator exported under EXPORTS
# (`sim --export`, EXPORTS/<gpu>/<spec>/<n>/) lifted (lift.py) and compiled by nvcc to a cubin for
# each target, and, unless --objects, run on this machine's GPU by the harness (main.cu's lift-run,
# built once) and its global memory compared with the simulator's byte for byte. One line per
# launch, then the counts; exits 1 when a launch fails to lift or compile, differs, or faults (a
# clock-reading kernel that differs is CLOCK-DEPENDENT, not a failure).
# usage: check.sh [--objects | --build-only | --run-only] [--arch sm_80,sm_90a] [--helpers N] EXPORTS
#   --objects: compile cubins for each target and the harness's object, run nothing (no GPU
#   needed); --build-only: build the harness and the cubins, run nothing; --run-only: run the ones
#   built. --arch: the targets (default: this machine's GPU's); NVCC: the compiler (default: nvcc
#   on PATH).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
nvcc=${NVCC:-nvcc}
flags="-O3 -std=c++17 -ftz=false -prec-div=true -prec-sqrt=true -fmad=false"

# One launch (`--one EXPORTS DIR ARCH...`): lifted, then compiled to a cubin for each target
# (DIR/<arch>.cubin); a failure is written to DIR/failed.
if [ "${1:-}" = "--one" ]; then
  e=$2; d=$3; shift 3
  rm -f "$d/failed"
  threads=$(awk '$1 == "block" { print $2 * $3 * $4 }' "$d/meta.txt")
  if ! python3 "$here/lift.py" --threads "$threads" "$d/kernel.jsonl" > "$d/lifted.cu" 2> "$d/lift.err"; then
    echo "not lifted: $(tail -1 "$d/lift.err" | cut -c1-300)" > "$d/failed"; exit 0
  fi
  for arch in "$@"; do
    # shellcheck disable=SC2086
    if ! "$nvcc" $flags -arch="$arch" -cubin -I"$e" -I"$d" -o "$d/$arch.cubin" "$d/lifted.cu" > "$d/compile-$arch.txt" 2>&1; then
      echo "not compiled for $arch: $(grep -m1 error "$d/compile-$arch.txt" | cut -c1-300)" > "$d/failed"; exit 0
    fi
  done
  exit 0
fi

build=1; run=1; objects=0; archs=""; helpers=""
while [ $# -gt 1 ]; do
  case "$1" in
    --objects) objects=1; run=0; shift ;;
    --build-only) run=0; shift ;;
    --run-only) build=0; shift ;;
    --arch) archs=$(echo "$2" | tr ',' ' '); shift 2 ;;
    --helpers) helpers="--helpers $2"; shift 2 ;;
    *) break ;;
  esac
done
exports=$(cd "$1" && pwd)
if [ -z "$archs" ]; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
  case "$cc" in 90|100|103) archs="sm_${cc}a" ;; *) archs="sm_$cc" ;; esac
fi
arch=${archs%% *}
if [ "$build" = 1 ]; then
  python3 "$here/lift.py" --tables "$exports/mufu_tables.cuh"
  cp "$here/rt.cuh" "$here/main.cu" "$exports/"
  # The harness: an object only, when nothing runs (no driver library to link against).
  if [ "$objects" = 1 ]; then out="$exports/lift-run.o"; link="-c"; else out="$exports/lift-run"; link="-lcuda"; fi
  # shellcheck disable=SC2086
  "$nvcc" $flags -arch="$arch" -I"$exports" -o "$out" "$exports/main.cu" $link > "$exports/compile-harness.txt" 2>&1 ||
    { echo "the harness not compiled: $(grep -m1 error "$exports/compile-harness.txt")"; exit 1; }
  find "$exports" -name kernel.jsonl -exec dirname {} \; | sort > "$exports/launches.txt"
  # shellcheck disable=SC2086
  xargs -P "$(nproc 2>/dev/null || echo 4)" -I{} sh "$0" --one "$exports" {} $archs < "$exports/launches.txt"
fi
ok=0; clock=0; bad=0
while read -r d; do
  name=${d#"$exports"/}
  if [ -f "$d/failed" ]; then echo "$name: $(cat "$d/failed")"; bad=$((bad + 1)); continue; fi
  if [ "$run" = 0 ]; then ok=$((ok + 1)); continue; fi
  set +e
  # shellcheck disable=SC2086
  r=$(timeout 900 "$exports/lift-run" "$d/$arch.cubin" "$d" 3 $helpers 2>&1); c=$?
  set -e
  echo "$name: $(echo "$r" | grep -E 'helper|BIT-EXACT|MISMATCH|CLOCK-DEPENDENT|differing|kernel time|fault flags|CUDA|fit' | tr '\n' ' ' | cut -c1-500) exit $c"
  case $c in 0) ok=$((ok + 1)) ;; 5) clock=$((clock + 1)) ;; *) bad=$((bad + 1)) ;; esac
done < "$exports/launches.txt"
if [ "$run" = 1 ]; then echo "bit-exact $ok, clock-dependent $clock, failed $bad"
else echo "lifted and compiled $ok, failed $bad"; fi
[ "$bad" -eq 0 ]
