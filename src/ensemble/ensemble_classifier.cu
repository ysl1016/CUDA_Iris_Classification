    #include "ensemble/ensemble_classifier.h"
    #include "utils/metrics_utils.h"
    #include <thrust/execution_policy.h>
    #include <chrono>
    /*
    Reasons why GPU implementation can enhance accuracy:
    1. Parallel processing enables more training iterations
    2. Batch processing provides more stable learning
    3. High-precision floating-point operations
    4. Ability to handle larger models and datasets efficiently
    5. Consistent results through deterministic operations
    */
    // Add constant for max epochs
    #define MAX_EPOCHS 100

    EnsembleClassifier::~EnsembleClassifier() noexcept {
        if (d_weights) cudaFree(d_weights);
        if (d_predictions) cudaFree(d_predictions);
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
        int* d_svm_pred = nullptr;
        int* d_nn_pred = nullptr;
        int* d_kmeans_pred = nullptr;
        
        try {
            CUDA_CHECK(cudaMalloc(&d_svm_pred, n_samples * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_nn_pred, n_samples * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_kmeans_pred, n_samples * sizeof(int)));
            
            // Get predictions
            svm.predict(features, d_svm_pred, n_samples);
            nn.predict(features, d_nn_pred, n_samples);
            kmeans.predict(features, n_samples, d_kmeans_pred);
            
            
            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
            
            // Copy individual predictions to combined array
            CUDA_CHECK(cudaMemcpy(d_predictions, d_svm_pred, n_samples * sizeof(int), 
                                cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_predictions + n_samples, d_nn_pred, n_samples * sizeof(int), 
                                cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_predictions + 2 * n_samples, d_kmeans_pred, n_samples * sizeof(int), 
                                cudaMemcpyDeviceToDevice));
            
            // Combine predictions using weighted voting
            weightedVoteKernel<<<grid_size, block_size>>>(
                d_predictions,
                d_weights,
                predictions,
                n_samples,
                n_classifiers,
                N_CLASSES
            );
            
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
        } catch (const std::runtime_error& e) {
            cudaFree(d_svm_pred);
            cudaFree(d_nn_pred);
            cudaFree(d_kmeans_pred);
            throw;
        }
        
        cudaFree(d_svm_pred);
        cudaFree(d_nn_pred);
        cudaFree(d_kmeans_pred);
    }

    float EnsembleClassifier::getAccuracy(const float* features, const int* labels, int n_samples) {
        int* predictions;
        CUDA_CHECK(cudaMalloc(&predictions, n_samples * sizeof(int)));
        
        auto start = std::chrono::high_resolution_clock::now();
        predict(features, predictions, n_samples);
        auto end = std::chrono::high_resolution_clock::now();
        float prediction_time = std::chrono::duration<float, std::milli>(end - start).count();
        
        float accuracy = MetricsUtils::calculateAccuracy(predictions, labels, n_samples);
        CUDA_CHECK(cudaFree(predictions));
        
        return accuracy;
    }

    void EnsembleClassifier::updateWeights(const float* features, const int* labels, int n_samples) {
        // Allocate host memory for accuracies
        float* accuracies = new float[n_classifiers];
        float total_accuracy = 0.0f;
        
        try {
            // Calculate accuracy for each classifier
            accuracies[0] = svm.getAccuracy(features, labels, n_samples);
            accuracies[1] = nn.getAccuracy(features, labels, n_samples);
            accuracies[2] = kmeans.getAccuracy(features, labels, n_samples);
            
            // Calculate total accuracy for normalization
            for (int i = 0; i < n_classifiers; i++) {
                total_accuracy += accuracies[i];
            }
            
            // Normalize weights
            if (total_accuracy > 0) {
                for (int i = 0; i < n_classifiers; i++) {
                    accuracies[i] /= total_accuracy;
                }
            } else {
                // If all classifiers perform poorly, use equal weights
                for (int i = 0; i < n_classifiers; i++) {
                    accuracies[i] = 1.0f / n_classifiers;
                }
            }
            
            // Update weights on device
            CUDA_CHECK(cudaMemcpy(d_weights, accuracies, n_classifiers * sizeof(float), 
                                cudaMemcpyHostToDevice));
        } catch (const std::runtime_error& e) {
            delete[] accuracies;
            throw std::runtime_error("Weight update failed: " + std::string(e.what()));
        }
        
        delete[] accuracies;
    }
