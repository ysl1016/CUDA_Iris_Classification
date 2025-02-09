#include "utils/metrics_utils.h"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <stdexcept>
#include <string>
#include <thrust/system/cuda/execution_policy.h>
#include <iostream>

namespace MetricsUtils {

struct CompareLabels {
    __host__ __device__
    int operator()(const thrust::tuple<const int&, const int&>& t) const {
        return thrust::get<0>(t) == thrust::get<1>(t) ? 1 : 0;
    }
};

// Calculate accuracy by comparing predictions with true labels
float calculateAccuracy(const int* predictions, const int* labels, int n_samples) {
    try {
        // 1. Debug info
        std::cout << "Starting accuracy calculation..." << std::endl;
        std::cout << "Number of samples: " << n_samples << std::endl;
        
        // 2. Create device vectors to manage memory automatically
        thrust::device_vector<int> d_pred(predictions, predictions + n_samples);
        thrust::device_vector<int> d_labels(labels, labels + n_samples);
        
        // 3. Get raw pointers for device vectors
        const int* raw_pred = thrust::raw_pointer_cast(d_pred.data());
        const int* raw_labels = thrust::raw_pointer_cast(d_labels.data());
        
        // 4. Synchronize and check for errors
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 5. Calculate accuracy
        int correct = thrust::transform_reduce(
            thrust::cuda::par,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(n_samples),
            [=] __host__ __device__ (int idx) -> int {
                return raw_pred[idx] == raw_labels[idx] ? 1 : 0;
            },
            0,
            thrust::plus<int>()
        );
        
        // 6. Final synchronization
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 7. Debug output
        std::cout << "Correct predictions: " << correct << "/" << n_samples << std::endl;
        
        return static_cast<float>(correct) / n_samples;
    } catch (const std::runtime_error& e) {
        std::cerr << "Detailed error in accuracy calculation: " << e.what() << std::endl;
        throw;
    }
}

// CUDA kernel for computing confusion matrix
__global__ void confusionMatrixKernel(const int* predictions,
                                     const int* labels,
                                     int* confusion_matrix,
                                     int n_samples,
                                     int n_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_samples) {
        int pred = predictions[idx];
        int true_label = labels[idx];
        // Atomic add to safely update confusion matrix in parallel
        atomicAdd(&confusion_matrix[true_label * n_classes + pred], 1);
    }
}

// Calculate confusion matrix for multi-class classification
void calculateConfusionMatrix(const int* predictions,
                            const int* labels,
                            int n_samples,
                            int n_classes,
                            int* confusion_matrix) {
    // Allocate and initialize device memory for confusion matrix
    int* d_confusion_matrix;
    cudaMalloc(&d_confusion_matrix, n_classes * n_classes * sizeof(int));
    cudaMemset(d_confusion_matrix, 0, n_classes * n_classes * sizeof(int));
    
    // Configure kernel execution parameters
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch confusion matrix kernel
    confusionMatrixKernel<<<grid_size, block_size>>>(
        predictions,
        labels,
        d_confusion_matrix,
        n_samples,
        n_classes
    );
    
    // Copy results back to host and cleanup
    cudaMemcpy(confusion_matrix, 
               d_confusion_matrix, 
               n_classes * n_classes * sizeof(int), 
               cudaMemcpyDeviceToHost);
    
    cudaFree(d_confusion_matrix);
}

// Calculate precision for a specific class
float calculatePrecision(const int* confusion_matrix, 
                        int class_idx, 
                        int n_classes) {
    // True positives are on the diagonal
    int true_positives = confusion_matrix[class_idx * n_classes + class_idx];
    int predicted_positives = 0;
    
    // Sum all predictions for this class (column sum)
    for (int i = 0; i < n_classes; i++) {
        predicted_positives += confusion_matrix[i * n_classes + class_idx];
    }
    
    // Return precision, handling division by zero
    return predicted_positives > 0 ? 
           static_cast<float>(true_positives) / predicted_positives : 0.0f;
}

// Calculate recall for a specific class
float calculateRecall(const int* confusion_matrix, 
                     int class_idx, 
                     int n_classes) {
    // True positives are on the diagonal
    int true_positives = confusion_matrix[class_idx * n_classes + class_idx];
    int actual_positives = 0;
    
    // Sum all actual instances of this class (row sum)
    for (int i = 0; i < n_classes; i++) {
        actual_positives += confusion_matrix[class_idx * n_classes + i];
    }
    
    // Return recall, handling division by zero
    return actual_positives > 0 ? 
           static_cast<float>(true_positives) / actual_positives : 0.0f;
}

// Calculate F1 score from precision and recall
float calculateF1Score(float precision, float recall) {
    // Return F1 score, handling division by zero
    return (precision + recall > 0.0f) ? 
           2.0f * precision * recall / (precision + recall) : 0.0f;
}

}
