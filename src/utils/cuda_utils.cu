#include "cuda_utils.h"
#include <iostream>

namespace CudaUtils {

void checkDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found!");
    }
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        
        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." 
                  << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " 
                  << deviceProp.totalGlobalMem / 1048576.0 << " MB" << std::endl;
        std::cout << "  Max threads per block: " 
                  << deviceProp.maxThreadsPerBlock << std::endl;
    }
}

template<typename T>
void allocateDeviceMemory(T** d_ptr, size_t size) {
    cudaError_t error = cudaMalloc(d_ptr, size);
    checkCudaError(error, __FILE__, __LINE__);
}

template<typename T>
void freeDeviceMemory(T* d_ptr) {
    if (d_ptr != nullptr) {
        cudaError_t error = cudaFree(d_ptr);
        checkCudaError(error, __FILE__, __LINE__);
    }
}

template<typename T>
void copyToDevice(T* d_dst, const T* h_src, size_t size) {
    cudaError_t error = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
    checkCudaError(error, __FILE__, __LINE__);
}

template<typename T>
void copyToHost(T* h_dst, const T* d_src, size_t size) {
    cudaError_t error = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
    checkCudaError(error, __FILE__, __LINE__);
}

void calculateGridSize(int n, int block_size, dim3& grid_size, dim3& block_dim) {
    block_dim.x = block_size;
    block_dim.y = 1;
    block_dim.z = 1;
    
    grid_size.x = (n + block_size - 1) / block_size;
    grid_size.y = 1;
    grid_size.z = 1;
}

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line 
                  << ": " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

void printMemoryUsage() {
    size_t free_byte;
    size_t total_byte;
    
    cudaError_t error = cudaMemGetInfo(&free_byte, &total_byte);
    checkCudaError(error, __FILE__, __LINE__);
    
    float free_mb = free_byte / 1024.0 / 1024.0;
    float total_mb = total_byte / 1024.0 / 1024.0;
    float used_mb = total_mb - free_mb;
    
    std::cout << "GPU Memory Usage:" << std::endl;
    std::cout << "  Total: " << total_mb << " MB" << std::endl;
    std::cout << "  Used:  " << used_mb << " MB" << std::endl;
    std::cout << "  Free:  " << free_mb << " MB" << std::endl;
}

// Explicit template instantiations for commonly used types
template void allocateDeviceMemory<float>(float**, size_t);
template void allocateDeviceMemory<int>(int**, size_t);
template void freeDeviceMemory<float>(float*);
template void freeDeviceMemory<int>(int*);
template void copyToDevice<float>(float*, const float*, size_t);
template void copyToDevice<int>(int*, const int*, size_t);
template void copyToHost<float>(float*, const float*, size_t);
template void copyToHost<int>(int*, const int*, size_t);

}
