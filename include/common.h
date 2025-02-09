#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>

// Constants
#define BLOCK_SIZE 256
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define MAX_EPOCHS 100
#define LEARNING_RATE 0.01f
#define CONVERGENCE_THRESHOLD 1e-6f

// Iris dataset constants
#define N_FEATURES 4
#define N_CLASSES 3

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cudaGetLastError(); \
            throw std::runtime_error(cudaGetErrorString(error)); \
        } \
    } while(0)

// Data structure for Iris dataset
struct IrisData {
    float* features;      // Input features
    int* labels;         // Class labels
    int n_samples;       // Number of samples
    int n_features;      // 
    int n_classes;       // 
    
    static constexpr int FEATURE_DIM = 4;
    static constexpr int CLASS_COUNT = 3;
    
    // Constructor
    IrisData() : features(nullptr), labels(nullptr), 
                 n_samples(0), n_features(FEATURE_DIM), n_classes(CLASS_COUNT) {}
};

// Performance metrics structure
struct PerformanceMetrics {
    float accuracy;
    float precision;
    float recall;
    float f1_score;
    float training_time;
    float inference_time;
    size_t memory_usage;
};
