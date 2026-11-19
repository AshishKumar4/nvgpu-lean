#!/bin/sh
# Check NV.Num's B200 tcgen05.mma datapaths against SF Tensor's model of them (docs/sources.md
# `sf-tensor`): fetch its repository at the pinned commit, draw cases with its model's results
# (tools/num/sftensor_vectors.py), run the check. N cases per datapath (default 2000).
set -eu
commit=d96b82180ac9041c09295f3a6a3c2c4e592ddce6
dir="${TMPDIR:-/tmp}/sf-tensor-tcgen05"
if [ ! -d "$dir" ]; then
  git clone -q https://github.com/sf-tensor/tcgen05 "$dir"
fi
git -C "$dir" checkout -q "$commit"
python3 tools/num/sftensor_vectors.py "$dir" "${1:-2000}" "$dir/cases.txt"
lake exe num-tests sftensor "$dir/cases.txt"
