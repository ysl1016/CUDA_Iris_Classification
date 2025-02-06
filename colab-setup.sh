#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

apt-get update
apt-get install -y cmake build-essential libgtest-dev python3-pip

cd /usr/src/gtest
cmake CMakeLists.txt
make
cp lib/*.a /usr/lib

pip install numpy pandas matplotlib seaborn

mkdir -p bin lib build data results

chmod +x install.sh run.sh
