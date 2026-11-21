#!/bin/sh
# A cuobjdump of another CUDA release in DIR (NVIDIA's redistributable, fetched once and checked
# against its digest), its path printed. The emulated device lists with them the cubins the
# toolkit's cuobjdump cannot: 12.8's reads the ELF format before CUDA 13's
# (NVGPU_LEAN_CUOBJDUMP_OLD), 13.5's what a newer nvJitLink than the toolkit writes
# (NVGPU_LEAN_CUOBJDUMP_NEW).
#   sh tools/cudadev/cuobjdump.sh 12.8|13.5 DIR
set -e
case $1 in
  12.8) v=12.8.90; sum=dbf41355a0f7a8aa491c1839a6c6ff8df02cdb6c6eefd66360543616c1b4980f ;;
  13.5) v=13.5.37; sum=cef2d02c05b3f6300ac8972ddb960feb899e614575476bf45dbbdf70c330d2ae ;;
  *) echo "cuobjdump.sh: no pinned cuobjdump $1" >&2; exit 1 ;;
esac
dir=$2
name=cuda_cuobjdump-linux-x86_64-$v-archive
bin="$dir/$name/bin/cuobjdump"
if [ ! -x "$bin" ]; then
  mkdir -p "$dir"
  curl -sfL -o "$dir/$name.tar.xz" \
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cuobjdump/linux-x86_64/$name.tar.xz"
  echo "$sum  $dir/$name.tar.xz" | sha256sum -c - > /dev/null
  tar xJf "$dir/$name.tar.xz" -C "$dir"
  rm "$dir/$name.tar.xz"
fi
echo "$bin"
