#include "ensemble/ensemble_classifier.h"
#include "utils/metrics_utils.h"
#include <thrust/execution_policy.h>

// Add constant for max epochs
#define MAX_EPOCHS 100

EnsembleClassifier::~EnsembleClassifier() {
    if (d_weights) CUDA_CHECK(cudaFree(d_weights));
    if (d_predictions) CUDA_CHECK(cudaFree(d_predictions));
}

__global__ void weightedVoteKernel(const int* individual_predictions,
                                 const float* weights,
                                 int* final_predictions,
                                 int n_samples,
                                 int n_classifiers,
                                 int n_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_samples) {
        float* class_scores = new float[n_classes]();
        
        // Accumulate weighted votes
        for (int i = 0; i < n_classifiers; i++) {
            int pred = individual_predictions[idx * n_classifiers + i];
            class_scores[pred] += weights[i];
        }
        
        // Find class with maximum score
        int max_class = 0;
        float max_score = class_scores[0];
        for (int i = 1; i < n_classes; i++) {
            if (class_scores[i] > max_score) {
                max_score = class_scores[i];
                max_class = i;
            }
        }
        
        final_predictions[idx] = max_class;
        delete[] class_scores;
    }
}

void EnsembleClassifier::train(const IrisData& data) {
    try {
        // Train individual classifiers
        svm.train(data);
        nn.train(data, MAX_EPOCHS);
        kmeans.train(data);
        
        // Initialize weights
        float initial_weight = 1.0f / n_classifiers;
        thrust::fill(thrust::device, 
                    thrust::device_pointer_cast(d_weights),
                    thrust::device_pointer_cast(d_weights + n_classifiers), 
                    initial_weight);
        
        // Update weights
        updateWeights(data.features, data.labels, data.n_samples);
        CUDA_CHECK(cudaGetLastError());
    }
    catch (const std::runtime_error& e) {
        throw std::runtime_error("Training failed: " + std::string(e.what()));
    }
}

void EnsembleClassifier::predict(const float* features, int* predictions, int n_samples) {
    // Get predictions from each classifier
    int* d_svm_pred;
    int* d_nn_pred;
    int* d_kmeans_pred;
    
    CUDA_CHECK(cudaMalloc(&d_svm_pred, n_samples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nn_pred, n_samples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_kmeans_pred, n_samples * sizeof(int)));
    
    svm.predict(features, d_svm_pred, n_samples);
    nn.predict(features, d_nn_pred, n_samples);
    kmeans.predict(features, n_samples, d_kmeans_pred);
    
    // Copy individual predictions to combined array
    CUDA_CHECK(cudaMemcpy(d_predictions, d_svm_pred, n_samples * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_predictions + n_samples, d_nn_pred, n_samples * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_predictions + 2 * n_samples, d_kmeans_pred, n_samples * sizeof(int), cudaMemcpyDeviceToDevice));
    
    // Combine predictions using weighted voting
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    weightedVoteKernel<<<grid_size, block_size>>>(
        d_predictions,
        d_weights,
        predictions,
        n_samples,
        n_classifiers,
        3  // n_classes for Iris dataset
    );
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_svm_pred));
    CUDA_CHECK(cudaFree(d_nn_pred));
    CUDA_CHECK(cudaFree(d_kmeans_pred));
}

void EnsembleClassifier::updateWeights(const float* features, const int* labels, int n_samples) {
    // Update weights based on individual classifier performance
    float* accuracies = new float[n_classifiers];
    accuracies[0] = svm.getAccuracy(features, labels, n_samples);
    accuracies[1] = nn.getAccuracy(features, labels, n_samples);
    accuracies[2] = kmeans.getAccuracy(features, labels, n_samples);
    
    // Copy accuracies to device and normalize them
    CUDA_CHECK(cudaMemcpy(d_weights, accuracies, n_classifiers * sizeof(float), 
                         cudaMemcpyHostToDevice));
    delete[] accuracies;
}
