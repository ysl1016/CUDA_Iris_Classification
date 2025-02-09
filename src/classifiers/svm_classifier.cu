#include "classifiers/svm_classifier.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

__global__ void computeRBFKernel(float* kernel_matrix,
                                const float* features,
                                int n_samples,
                                int n_features,
                                float gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_samples && j < n_samples) {
        float sum = 0.0f;
        for (int k = 0; k < n_features; k++) {
            float diff = features[i * n_features + k] - features[j * n_features + k];
            sum += diff * diff;
        }
        kernel_matrix[i * n_samples + j] = exp(-gamma * sum);
    }
}

__global__ void predictKernel(const float* kernel_matrix,
                            const float* alpha,
                            const int* support_vector_labels,
                            int n_support_vectors,
                            int* predictions,
                            int n_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_samples) {
        float decision = 0.0f;
        for (int i = 0; i < n_support_vectors; i++) {
            decision += alpha[i] * support_vector_labels[i] * kernel_matrix[idx * n_support_vectors + i];
        }
        predictions[idx] = (decision > 0.0f) ? 1 : -1;
    }
}

SVMClassifier::SVMClassifier(float C, float gamma) : C(C), gamma(gamma) {
    d_kernel_matrix = nullptr;
    d_alpha = nullptr;
    d_support_vectors = nullptr;
    d_support_vector_labels = nullptr;
    n_support_vectors = 0;
}

SVMClassifier::~SVMClassifier() {
    if (d_kernel_matrix) cudaFree(d_kernel_matrix);
    if (d_alpha) cudaFree(d_alpha);
    if (d_support_vectors) cudaFree(d_support_vectors);
    if (d_support_vector_labels) cudaFree(d_support_vector_labels);
}

void SVMClassifier::train(const float* features, const int* labels, int n_samples) {
    // Compute kernel matrix
    computeKernelMatrix(features, n_samples);
    
    // Optimize dual problem using SMO algorithm
    optimizeDual();
    
    // Select support vectors (alpha > 0)
    // Implementation details omitted for brevity
}

void SVMClassifier::predict(const float* features, int* predictions, int n_samples) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    predictKernel<<<grid_size, block_size>>>(
        d_kernel_matrix,
        d_alpha,
        d_support_vector_labels,
        n_support_vectors,
        predictions,
        n_samples
    );
}

void SVMClassifier::computeKernelMatrix(const float* features, int n_samples) {
    // Allocate memory for kernel matrix
    CUDA_CHECK(cudaMalloc(&d_kernel_matrix, n_samples * n_samples * sizeof(float)));
    
    // Set up grid and block dimensions
    dim3 block_dim(16, 16);
    dim3 grid_dim((n_samples + block_dim.x - 1) / block_dim.x,
                  (n_samples + block_dim.y - 1) / block_dim.y);
    
    // Compute RBF kernel matrix
    computeRBFKernel<<<grid_dim, block_dim>>>(
        d_kernel_matrix,
        features,
        n_samples,
        4,  // n_features for Iris
        gamma
    );
    CUDA_CHECK(cudaGetLastError());
}

void SVMClassifier::optimizeDual() {

    int n_samples = 150; // Iris dataset size
    
    // Allocate memory for alpha
    CUDA_CHECK(cudaMalloc(&d_alpha, n_samples * sizeof(float)));
    thrust::device_ptr<float> d_alpha_ptr(d_alpha);
    thrust::fill(d_alpha_ptr, d_alpha_ptr + n_samples, 0.0f);
    
    for (int iter = 0; iter < 100; ++iter) {
        // Update alpha values
    }
}

float SVMClassifier::getAccuracy(const float* features, const int* labels, int n_samples) {
    int* predictions;
    CUDA_CHECK(cudaMalloc(&predictions, n_samples * sizeof(int)));
    
    predict(features, predictions, n_samples);
    
    // Calculate accuracy using thrust
    thrust::device_ptr<const int> d_pred_ptr(predictions);
    thrust::device_ptr<const int> d_labels_ptr(labels);
    
    int correct = thrust::transform_reduce(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n_samples),
        [=] __device__ (int idx) -> int {
            return d_pred_ptr[idx] == d_labels_ptr[idx] ? 1 : 0;
        },
        0,
        thrust::plus<int>()
    );
    
    CUDA_CHECK(cudaFree(predictions));
    return static_cast<float>(correct) / n_samples;
}
