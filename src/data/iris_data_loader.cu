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

bool IrisDataLoader::loadData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::vector<float> features;
    std::vector<int> labels;
    std::string line;

    // Read data from CSV file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        
        // Read features
        for (int i = 0; i < 4; i++) {
            std::getline(ss, value, ',');
            features.push_back(std::stof(value));
        }
        
        // Read label
        std::getline(ss, value, ',');
        labels.push_back(std::stoi(value));
    }

    data.n_samples = labels.size();
    allocateMemory(data.n_samples);

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(data.features, features.data(), 
                         features.size() * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.labels, labels.data(), 
                         labels.size() * sizeof(int), 
                         cudaMemcpyHostToDevice));

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
