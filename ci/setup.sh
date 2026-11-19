#!/bin/sh
# As root, once per CI environment: system packages (g++ is nvcc's host compiler; a JDK and lark
# run the Alloy cross-check, ci/rows/alloy.sh), and elan for the `ci` user. The Lean toolchain itself is the checkout's `lean-toolchain`, installed by
# ci/install.sh.
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends curl ca-certificates git python3 python3-venv time g++ \
  default-jdk-headless python3-lark
ELAN_VERSION=4.2.4
ELAN_SHA256=42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63
curl -fsSL -o /tmp/elan.tar.gz "https://github.com/leanprover/elan/releases/download/v${ELAN_VERSION}/elan-x86_64-unknown-linux-gnu.tar.gz"
echo "${ELAN_SHA256}  /tmp/elan.tar.gz" | sha256sum -c
tar xzf /tmp/elan.tar.gz -C /tmp
ELAN_HOME=/home/ci/.elan /tmp/elan-init -y --no-modify-path --default-toolchain none
chown -R ci: /home/ci/.elan
rm -f /tmp/elan.tar.gz /tmp/elan-init
