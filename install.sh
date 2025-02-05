#!/bin/bash

# Check for CUDA installation
if ! command -v nvcc &>/dev/null; then
    echo "CUDA toolkit not found. Please install CUDA toolkit first."
    exit 1
fi

# Create necessary directories
mkdir -p bin lib build

# Install required packages
if command -v apt-get &>/dev/null; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y cmake build-essential python3-pip
elif command -v yum &>/dev/null; then
    # CentOS/RHEL
    sudo yum update
    sudo yum install -y cmake gcc-c++ python3-pip
fi

# Install Python packages for visualization
pip3 install numpy pandas matplotlib seaborn

# Download Iris dataset
python3 scripts/download_iris.py

# Install Doxygen if not present
if ! command -v doxygen &>/dev/null; then
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y doxygen
    elif command -v yum &>/dev/null; then
        sudo yum install -y doxygen
    fi
fi

# Generate documentation
doxygen Doxyfile

# Build project with tests
mkdir -p build
cd build
cmake ..
make -j4

echo "Installation complete!"
echo "Executables can be found in the 'bin' directory"
echo "Libraries can be found in the 'lib' directory"
