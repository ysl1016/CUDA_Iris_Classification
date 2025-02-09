#include "data/iris_data_loader.h"
#include <fstream>
#include <sstream>

bool IrisDataLoader::loadData(IrisData& data) {
    std::vector<float> features;
    std::vector<int> labels;
    
    if (!loadFromFile(features, labels)) {
        return false;
    }
    
    int n_samples = labels.size();
    data.n_samples = n_samples;
    
    if (!allocateMemory(data, n_samples)) {
        return false;
    }
    
    // Copy data to device
    cudaError_t error;
    error = cudaMemcpy(data.features, features.data(), 
                      features.size() * sizeof(float), 
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        freeMemory(data);
        return false;
    }
    
    error = cudaMemcpy(data.labels, labels.data(), 
                      labels.size() * sizeof(int), 
                      cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        freeMemory(data);
        return false;
    }
    
    return true;
}

bool IrisDataLoader::allocateMemory(IrisData& data, int n_samples) {
    cudaError_t error;
    
    error = cudaMalloc(&data.features, n_samples * N_FEATURES * sizeof(float));
    if (error != cudaSuccess) {
        return false;
    }
    
    error = cudaMalloc(&data.labels, n_samples * sizeof(int));
    if (error != cudaSuccess) {
        if (data.features) cudaFree(data.features);
        return false;
    }
    
    return true;
}

void IrisDataLoader::freeMemory(IrisData& data) {
    if (data.features) {
        cudaFree(data.features);
        data.features = nullptr;
    }
    if (data.labels) {
        cudaFree(data.labels);
        data.labels = nullptr;
    }
}
