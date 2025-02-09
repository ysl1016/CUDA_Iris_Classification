#include "data/iris_data_loader.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>

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

bool IrisDataLoader::loadFromFile(std::vector<float>& features, std::vector<int>& labels) {
    const char* url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
    
    std::string command = "wget -O data/iris.data " + std::string(url);
    int result = system(command.c_str());
    
    if (result != 0) {
        std::cerr << "Failed to download iris dataset" << std::endl;
        return false;
    }
    
    std::ifstream file("data/iris.data");
    if (!file.is_open()) {
        std::cerr << "Could not open iris.data" << std::endl;
        return false;
    }
    
    features.clear();
    labels.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;
        
        // Read 4 feature values
        for (int i = 0; i < 4; i++) {
            std::getline(ss, value, ',');
            row.push_back(std::stof(value));
        }
        
        // Read class label
        std::getline(ss, value);
        int label;
        if (value == "Iris-setosa") label = 0;
        else if (value == "Iris-versicolor") label = 1;
        else if (value == "Iris-virginica") label = 2;
        else continue;
        
        features.insert(features.end(), row.begin(), row.end());
        labels.push_back(label);
    }
    
    file.close();
    return !features.empty();
}
