#pragma once
#include "common.h"
#include <string>

namespace CudaUtils {
    // Check CUDA device information and properties
    void checkDeviceProperties();
    
    // Memory management helper functions
    template<typename T>
    void allocateDeviceMemory(T** d_ptr, size_t size);
    
    template<typename T>
    void freeDeviceMemory(T* d_ptr);
    
    template<typename T>
    void copyToDevice(T* d_dst, const T* h_src, size_t size);
    
    template<typename T>
    void copyToHost(T* h_dst, const T* d_src, size_t size);
    
    // Calculate CUDA kernel execution configuration
    void calculateGridSize(int n, int block_size, dim3& grid_size, dim3& block_dim);
    
    // CUDA error checking and logging
    void checkCudaError(cudaError_t error, const char* file, int line);
    
    // Monitor GPU memory usage
    void printMemoryUsage();
}
