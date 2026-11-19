#!/bin/sh
# As the `ci` user in the checkout, once per environment: the pinned Lean toolchain and the pinned
# CUDA compiler tools (tools/cuda/install.sh) in ~/cuda, and the project's Lean build.
set -eu
"$HOME/.elan/bin/elan" toolchain install "$(cut -d: -f2 lean-toolchain)"
sh tools/cuda/install.sh "$HOME/cuda"
# The project built once into the environment: a task's `lake build` then rebuilds only the
# modules its commit changed (Lake compares source hashes), not the whole library. A failure here
# only costs the cache.
PATH="$HOME/.elan/bin:$PATH" lake build > /tmp/prebuild.log 2>&1 || tail -20 /tmp/prebuild.log
