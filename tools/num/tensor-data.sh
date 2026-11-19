#!/bin/sh
# Check NV.Num's tensor-core models against Khattak and Mikaitis's A100, H100/H200 and B200
# measurements
# (docs/sources.md `khattak2025`): fetch their repository at the pinned commit, run the check.
set -eu
commit=79143bd29a4a71dc7c2ed47741b63572498b17f6
dir="${TMPDIR:-/tmp}/matlab-tensor-core"
if [ ! -d "$dir" ]; then
  git clone -q https://github.com/north-numerical-computing/MATLAB-tensor-core "$dir"
fi
git -C "$dir" checkout -q "$commit"
lake exe num-tests tensor "$dir/model_validation"
