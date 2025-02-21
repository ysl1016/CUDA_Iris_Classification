    #include "ensemble/ensemble_classifier.h"
    #include "utils/metrics_utils.h"
    #include <thrust/execution_policy.h>
    #include <thrust/transform_reduce.h>
    #include <thrust/functional.h>
    #include <thrust/iterator/counting_iterator.h>
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

    EnsembleClassifier::EnsembleClassifier() : nn(N_FEATURES, 8, N_CLASSES), kmeans(N_CLASSES) {
        CUDA_CHECK(cudaMalloc(&d_weights, n_classifiers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_predictions, MAX_SAMPLES * n_classifiers * sizeof(int)));
    }

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
        
        // shared memory 
        extern __shared__ float shared_scores[];
        float* class_scores = &shared_scores[threadIdx.x * n_classes];
        
        if (idx < n_samples) {
            // Initialize scores to 0
            for (int i = 0; i < n_classes; i++) {
                class_scores[i] = 0.0f;
            }
            
            // Accumulate weighted votes
            for (int i = 0; i < n_classifiers; i++) {
                int pred = individual_predictions[i * n_samples + idx];
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
        }
    }

    void EnsembleClassifier::train(const IrisData& data) {
        try {
            std::cout << "Training SVM classifier..." << std::endl;
            svm.train(data.features, data.labels, data.n_samples);
            
            std::cout << "Training Neural Network..." << std::endl;
            nn.train(data.features, data.labels, data.n_samples, MAX_EPOCHS);
            
            std::cout << "Training KMeans classifier..." << std::endl;
            kmeans.train(data);
            
            std::cout << "Initializing weights..." << std::endl;
            float initial_weight = 1.0f / n_classifiers;
            thrust::fill(thrust::device, 
                        thrust::device_pointer_cast(d_weights),
                        thrust::device_pointer_cast(d_weights + n_classifiers), 
                        initial_weight);
            
            std::cout << "Updating weights..." << std::endl;
            updateWeights(data.features, data.labels, data.n_samples);
            
            CUDA_CHECK(cudaGetLastError());
            std::cout << "Training completed successfully" << std::endl;
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
            // Allocate memory
            cudaMalloc(&d_svm_pred, n_samples * sizeof(int));
            cudaMalloc(&d_nn_pred, n_samples * sizeof(int));
            cudaMalloc(&d_kmeans_pred, n_samples * sizeof(int));
            
            // Get predictions
            svm.predict(features, d_svm_pred, n_samples);
            nn.predict(features, d_nn_pred, n_samples);
            kmeans.predict(features, n_samples, d_kmeans_pred);
            
            // Copy predictions and combine
            cudaMemcpy(d_predictions, d_svm_pred, n_samples * sizeof(int), 
                      cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_predictions + n_samples, d_nn_pred, n_samples * sizeof(int), 
                      cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_predictions + (2 * n_samples), d_kmeans_pred, n_samples * sizeof(int), 
                      cudaMemcpyDeviceToDevice);
            
            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
            
            size_t shared_mem_size = BLOCK_SIZE * N_CLASSES * sizeof(float);

            weightedVoteKernel<<<grid_size, block_size, shared_mem_size>>>(
                d_predictions,
                d_weights,
                predictions,
                n_samples,
                n_classifiers,
                N_CLASSES
            );
            
            cudaDeviceSynchronize();
        }
        catch (const std::runtime_error& e) {
            if (d_svm_pred) cudaFree(d_svm_pred);
            if (d_nn_pred) cudaFree(d_nn_pred);
            if (d_kmeans_pred) cudaFree(d_kmeans_pred);
            throw;
        }
        
        if (d_svm_pred) cudaFree(d_svm_pred);
        if (d_nn_pred) cudaFree(d_nn_pred);
        if (d_kmeans_pred) cudaFree(d_kmeans_pred);
    }

    float EnsembleClassifier::getAccuracy(const float* features, const int* labels, int n_samples) {
        // Allocate device memory for predictions
        thrust::device_vector<int> d_predictions(n_samples);
        
        try {
            // Get predictions
            predict(features, thrust::raw_pointer_cast(d_predictions.data()), n_samples);
            
            // Calculate accuracy
            float accuracy = MetricsUtils::calculateAccuracy(
                thrust::raw_pointer_cast(d_predictions.data()),
                labels,
                n_samples
            );
            
            return accuracy;
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("Accuracy calculation failed: " + std::string(e.what()));
        }
    }

    void EnsembleClassifier::updateWeights(const float* features, const int* labels, int n_samples) {
        // Allocate device memory for predictions
        int *d_svm_pred, *d_nn_pred, *d_kmeans_pred;
        CUDA_CHECK(cudaMalloc(&d_svm_pred, n_samples * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nn_pred, n_samples * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_kmeans_pred, n_samples * sizeof(int)));
        
        try {
            // Get predictions from each classifier
            svm.predict(features, d_svm_pred, n_samples);
            nn.predict(features, d_nn_pred, n_samples);
            kmeans.predict(features, n_samples, d_kmeans_pred);
            
            // Calculate accuracies using MetricsUtils
            float accuracies[n_classifiers];
            accuracies[0] = MetricsUtils::calculateAccuracy(d_svm_pred, labels, n_samples);
            accuracies[1] = MetricsUtils::calculateAccuracy(d_nn_pred, labels, n_samples);
            accuracies[2] = MetricsUtils::calculateAccuracy(d_kmeans_pred, labels, n_samples);
            
            float total_accuracy = 0.0f;
            for (int i = 0; i < n_classifiers; i++) {
                total_accuracy += accuracies[i];
            }
            
            // Normalize weights
            if (total_accuracy > 0) {
                for (int i = 0; i < n_classifiers; i++) {
                    accuracies[i] /= total_accuracy;
                }
            } else {
                for (int i = 0; i < n_classifiers; i++) {
                    accuracies[i] = 1.0f / n_classifiers;
                }
            }
            
            // Update weights on device
            CUDA_CHECK(cudaMemcpy(d_weights, accuracies, n_classifiers * sizeof(float), 
                                cudaMemcpyHostToDevice));
                            
            // Cleanup
            CUDA_CHECK(cudaFree(d_svm_pred));
            CUDA_CHECK(cudaFree(d_nn_pred));
            CUDA_CHECK(cudaFree(d_kmeans_pred));
        }
        catch (const std::runtime_error& e) {
            if (d_svm_pred) cudaFree(d_svm_pred);
            if (d_nn_pred) cudaFree(d_nn_pred);
            if (d_kmeans_pred) cudaFree(d_kmeans_pred);
            throw std::runtime_error("Weight update failed: " + std::string(e.what()));
        }
    }

    bool EnsembleClassifier::init() {
        try {
            if (d_weights == nullptr || d_predictions == nullptr) {
                // Allocate memory for weights and predictions
                CUDA_CHECK(cudaMalloc(&d_weights, n_classifiers * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_predictions, MAX_SAMPLES * n_classifiers * sizeof(int)));
                
                // Initialize weights equally
                float initial_weight = 1.0f / n_classifiers;
                thrust::fill(thrust::device, 
                            thrust::device_pointer_cast(d_weights),
                            thrust::device_pointer_cast(d_weights + n_classifiers), 
                            initial_weight);
            }
            return true;
        }
        catch (const std::runtime_error& e) {
            cleanup();
            return false;
        }
    }

    void EnsembleClassifier::cleanup() {
        if (d_weights) {
            cudaFree(d_weights);
            d_weights = nullptr;
        }
        if (d_predictions) {
            cudaFree(d_predictions);
            d_predictions = nullptr;
        }
    }
