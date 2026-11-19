#!/bin/sh
# The pinned CUDA compiler tools, from NVIDIA's PyPI wheels; no GPU and no driver needed.
# - ptxas, nvcc, cicc (nvvm), and the runtime and crt headers from CUDA 13.0.88.
# - CCCL 13.0.85 (libcu++, CUB, Thrust): the newest 13.0 wheel.
# - cuobjdump and nvdisasm 13.2.51: the oldest wheels that ship them.
# Usage: tools/cuda/install.sh DEST. The tools land in DEST/bin, headers in DEST/include (CCCL in
# DEST/include/cccl); nvcc finds both through DEST/bin/nvcc.profile and needs a host g++.
# The script itself needs curl, sha256sum and python3.
set -eu
dest=$1
pypi=https://files.pythonhosted.org/packages
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fetch() { # path-under-pypi sha256
  f="$tmp/$(basename "$1")"
  curl -fsSL --retry 6 --retry-delay 10 --retry-all-errors -o "$f" "$pypi/$1"
  echo "$2  $f" | sha256sum -c --quiet
  python3 -m zipfile -e "$f" "$tmp/x"
}
fetch 71/8b/a546c12881fffeba927d810598987df25d74b8b241788c7db8dfc93b0173/nvidia_cuda_nvcc-13.0.88-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
  56fe502eb77625a12f25172caa3cdddb4e4c8ba2c8c17dba44b164761b380f03
fetch 15/b0/ee41e6d1108d959b5097163e7190c2d0f7857dea75606ce358f0275891b4/nvidia_nvvm-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl \
  c5f41ffeb6466944a026dfa5317d7d85355c119bbec279205d22f1869d1054e0
fetch 05/69/a1ec4d9f0747d85964206feb47bf359f48a2d6af74c0add8abba6efe3dda/nvidia_cuda_crt-13.0.88-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
  2c8043c7c9e02492716426e9919fc78d2c5b3b2a7a768a88e952676b08aa55a4
fetch b1/1d/9ea7b0d420efcae3c546feb3ca14334c0d710ac19c156ac48fce07c3d6ae/nvidia_cuda_runtime-13.0.88-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
  422a38f7c739ecbdd3a4b6ec7ed0a1d74db43d8e9bc26685f3d845638922b178
fetch ab/fb/0384bb2129bed6b1b39f8e44471c615ab5ab29b7e55817538a1d390d8f84/nvidia_cuda_cccl-13.0.85-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
  e0da7ad981f3a8aff08241b5bfc1af868742a63e2762f53a5171c492ef242649
fetch a9/2c/af21e3b8eecbf1e9c373967fe517fd5236d061646664a98277de7d21c420/nvidia_cuda_cuobjdump-13.2.51-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
  95c69d3ddf0c2c1f535ec44c3433a66aeee19f588d9d22f73356e20b25f12b50
fetch ad/34/f19b4dd50158e7dc18c55654f844d87bf188d2a2b230374620e6b126a6c9/nvidia_cuda_nvdisasm-13.2.51-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
  5a7acf77c55d6d2911ed6155d4d5bc156ffadf9df7a68a89b0c1155f4b13688c
fetch 82/2a/5bdda446d0b5daa832a86b96e626d6ad4bb869fcda4483b2aa0cd970e651/nvidia_nvml_dev-13.0.87-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl \
  082871368275cf6620fc900f55d8eaf2563b263a38a00a5168e9650bce963ba1
mkdir -p "$dest"
rm -rf "$tmp/x/nvidia/cu13/.metadata"
cp -R "$tmp/x/nvidia/cu13/." "$dest/"
# nvml.h wherever its wheel keeps it.
find "$tmp/x" -name nvml.h -exec cp {} "$dest/include/" \;
chmod -R a+rX "$dest"
find "$dest/bin" "$dest/nvvm/bin" -type f -exec chmod a+x {} +
"$dest/bin/ptxas" --version | tail -1
