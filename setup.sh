#!/bin/bash
set -e

git config --global user.email "paul.plays.a.pun@gmail.com"
git config --global user.name  "better-call-paul"

if [ -d /workspace ]; then
  DEV_DIR=/workspace/devlibs
else
  DEV_DIR=~/devlibs
fi
mkdir -p "$DEV_DIR"
cd "$DEV_DIR"

apt-get update
apt-get install -y curl wget gnupg lsb-release build-essential

rm -f /etc/apt/sources.list.d/cuda-ubuntu*.list
rm -f /etc/apt/sources.list.d/cuda.list

distribution=$(source /etc/os-release && echo ${ID}${VERSION_ID//./})
KEYRING_DEB=cuda-keyring_1.1-1_all.deb
curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${distribution}/x86_64/${KEYRING_DEB}" \
  -o "/tmp/${KEYRING_DEB}"
dpkg -i "/tmp/${KEYRING_DEB}"
rm "/tmp/${KEYRING_DEB}"

apt-get update

PKG=$(apt-cache search '^nsight-compute-[0-9]' | awk '{print $1}' | sort -V | tail -n1)
apt-get install -y "$PKG"

apt-get install -y cmake git

if [ ! -d FlameGraph ]; then
  git clone https://github.com/brendangregg/FlameGraph.git
fi

if [ ! -d nvbench ]; then
  git clone https://github.com/NVIDIA/nvbench.git
  cd nvbench
  mkdir -p build && cd build
  cmake -DNVBench_ENABLE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release ..
  make -j4
  cd "$DEV_DIR"
fi

apt-get clean
rm -rf /var/lib/apt/lists/*

echo "Setup complete. Dev files in $DEV_DIR"
