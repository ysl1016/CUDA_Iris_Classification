#include "utils/cuda_utils.h"
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
T* allocateDevice(size_t size) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    return ptr;
}

template<typename T>
void freeDevice(T* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

template<typename T>
void copyToDevice(T* d_dst, const T* h_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copyToHost(T* h_dst, const T* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size * sizeof(T), cudaMemcpyDeviceToHost));
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
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, 
                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
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

void printDeviceProperties() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions: " << prop.maxThreadsDim[0] << "x" 
              << prop.maxThreadsDim[1] << "x" << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid dimensions: " << prop.maxGridSize[0] << "x" 
              << prop.maxGridSize[1] << "x" << prop.maxGridSize[2] << std::endl;
}

template<typename T>
void copyHostToDevice(T* d_dest, const T* h_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dest, h_src, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copyDeviceToHost(T* h_dest, const T* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_dest, d_src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

// Explicit template instantiations
template float* allocateDevice<float>(size_t);
template int* allocateDevice<int>(size_t);
template void freeDevice<float>(float*);
template void freeDevice<int>(int*);
template void copyToDevice<float>(float*, const float*, size_t);
template void copyToDevice<int>(int*, const int*, size_t);
template void copyToHost<float>(float*, const float*, size_t);
template void copyToHost<int>(int*, const int*, size_t);
template void copyHostToDevice<float>(float*, const float*, size_t);
template void copyHostToDevice<int>(int*, const int*, size_t);
template void copyDeviceToHost<float>(float*, const float*, size_t);
template void copyDeviceToHost<int>(int*, const int*, size_t);

}
