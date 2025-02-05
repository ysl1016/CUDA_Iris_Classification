#include "svm_classifier.h"

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

void SVMClassifier::train(const IrisData& data) {
    // Compute kernel matrix
    computeKernelMatrix(data.features, data.n_samples);
    
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
