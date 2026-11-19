#!/bin/sh
# Every pair of binary16 and of bfloat16 operands through add, sub, mul and div in each IEEE
# rounding mode, NV.Num against a host reference (tools/num/exhaustive_ref.c), compared by hashes.
# Usage: tools/num/exhaustive.sh SHARD SHARDS (SHARD from 0); the full check is every shard, e.g.
#   for i in $(seq 0 63); do sh tools/num/exhaustive.sh $i 64; done
set -eu
shard=$1; shards=$2
cc -std=c11 -O2 -frounding-math -ffp-contract=off -o /tmp/exhaustive_ref tools/num/exhaustive_ref.c -lm
lake build num-tests > /dev/null
for f in f16 bf16; do
  /tmp/exhaustive_ref "$f" "$shard" "$shards" > /tmp/ref.txt
  lake exe num-tests exhaustive "$f" "$shard" "$shards" > /tmp/got.txt
  if cmp -s /tmp/ref.txt /tmp/got.txt; then echo "ok   exhaustive $f shard $shard/$shards"
  else echo "FAIL exhaustive $f shard $shard/$shards"; diff /tmp/ref.txt /tmp/got.txt; exit 1; fi
done
