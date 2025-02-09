#include "data/iris_data_loader.h"
#include <fstream>
#include <sstream>

IrisDataLoader::IrisDataLoader() {
    data.features = nullptr;
    data.labels = nullptr;
    data.n_samples = 0;
    data.n_features = 4;
    data.n_classes = 3;
}

IrisDataLoader::~IrisDataLoader() {
    freeMemory();
}

bool IrisDataLoader::loadData(IrisData& data) {
    std::vector<float> features;
    std::vector<int> labels;
    
    if (!loadFromFile(features, labels)) {
        return false;
    }
    
    data.n_samples = labels.size();
    data.n_features = 4;
    data.n_classes = 3;
    
    // Allocate device memory
    if (cudaMalloc(&data.features, features.size() * sizeof(float)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&data.labels, labels.size() * sizeof(int)) != cudaSuccess) {
        cudaFree(data.features);
        return false;
    }
    
    // Copy data to device
    if (cudaMemcpy(data.features, features.data(), 
                   features.size() * sizeof(float), 
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(data.features);
        cudaFree(data.labels);
        return false;
    }
    
    if (cudaMemcpy(data.labels, labels.data(), 
                   labels.size() * sizeof(int), 
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(data.features);
        cudaFree(data.labels);
        return false;
    }
    
    return true;
}

void IrisDataLoader::allocateMemory(int n_samples) {
    CUDA_CHECK(cudaMalloc(&data.features, n_samples * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.labels, n_samples * sizeof(int)));
}

void IrisDataLoader::freeMemory() {
    if (data.features) CUDA_CHECK(cudaFree(data.features));
    if (data.labels) CUDA_CHECK(cudaFree(data.labels));
}
