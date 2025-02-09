#include <iostream>
#include <iomanip>
#include <chrono>
#include "data/iris_data_loader.h"
#include "preprocessing/data_preprocessor.h"
#include "ensemble/ensemble_classifier.h"
#include "utils/metrics_utils.h"
/*
Reasons why GPU implementation can enhance accuracy:
1. Parallel processing enables more training iterations
2. Batch processing provides more stable learning
3. High-precision floating-point operations
4. Ability to handle larger models and datasets efficiently
5. Consistent results through deterministic operations
*/

void printResults(const std::string& classifier_name, 
                 float accuracy, 
                 float training_time, 
                 float prediction_time) {
    std::cout << std::setw(15) << classifier_name << " | "
              << std::setw(10) << std::fixed << std::setprecision(2) 
              << accuracy * 100 << "% | "
              << std::setw(8) << training_time << "ms | "
              << std::setw(8) << prediction_time << "ms" << std::endl;
}

void measureClassifierPerformance(const char* name, 
                                const IrisData& train_data,
                                const IrisData& test_data,
                                std::function<void(const IrisData&)> train_fn,
                                std::function<float(const float*, const int*, int)> predict_fn) {
    auto train_start = std::chrono::high_resolution_clock::now();
    train_fn(train_data);
    auto train_end = std::chrono::high_resolution_clock::now();
    float train_time = std::chrono::duration<float, std::milli>(train_end - train_start).count();

    auto predict_start = std::chrono::high_resolution_clock::now();
    float accuracy = predict_fn(test_data.features, test_data.labels, test_data.n_samples);
    auto predict_end = std::chrono::high_resolution_clock::now();
    float predict_time = std::chrono::duration<float, std::milli>(predict_end - predict_start).count();

    printResults(name, accuracy, train_time, predict_time);
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string data_path = "data/iris.csv";
    if (argc > 1) {
        data_path = argv[1];
    }

    try {
        // Load data
        std::cout << "Loading data from " << data_path << "..." << std::endl;
        IrisDataLoader loader;
        if (!loader.loadData(data_path)) {
            throw std::runtime_error("Failed to load data");
        }

        // Preprocess data
        std::cout << "Preprocessing data..." << std::endl;
        DataPreprocessor preprocessor;
        IrisData& data = loader.getData();
        preprocessor.normalizeFeatures(data);

        // Split data into train and test sets
        IrisData train_data, test_data;
        preprocessor.splitData(data, train_data, test_data, 0.2f);

        // Initialize classifiers
        SVMClassifier svm;
        NeuralNetwork nn(4, 8, 3);  // 4 inputs, 8 hidden, 3 outputs
        KMeansClassifier kmeans(3);  // 3 clusters
        EnsembleClassifier ensemble;

        std::cout << "\nTraining and evaluating classifiers...\n" << std::endl;
        std::cout << std::setw(15) << "Classifier" << " | "
                 << std::setw(10) << "Accuracy" << " | "
                 << std::setw(8) << "Train ms" << " | "
                 << std::setw(8) << "Pred ms" << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        // Initialize classifiers and measure performance
        auto start = std::chrono::high_resolution_clock::now();

        // Train classifiers
        svm.train(train_data);
        nn.train(train_data, 100);  // 100 epochs
        kmeans.train(train_data);

        auto end = std::chrono::high_resolution_clock::now();
        float training_time = std::chrono::duration<float, std::milli>(end - start).count();

        // Make predictions and measure accuracy
        start = std::chrono::high_resolution_clock::now();
        int* predictions;
        CUDA_CHECK(cudaMalloc(&predictions, test_data.n_samples * sizeof(int)));

        // Get accuracy for each classifier
        float svm_accuracy = svm.getAccuracy(test_data.features, test_data.labels, test_data.n_samples);
        float nn_accuracy = nn.getAccuracy(test_data.features, test_data.labels, test_data.n_samples);

        // For KMeans, we need to predict first then calculate accuracy
        kmeans.predict(test_data.features, test_data.n_samples, predictions);
        float kmeans_accuracy = kmeans.accuracy(predictions, test_data.labels, test_data.n_samples);

        end = std::chrono::high_resolution_clock::now();
        float prediction_time = std::chrono::duration<float, std::milli>(end - start).count();

        // Print results
        std::cout << "\nClassifier Results:\n";
        std::cout << std::string(45, '-') << std::endl;
        std::cout << std::setw(15) << "Classifier" << " | "
                  << std::setw(10) << "Accuracy" << " | "
                  << std::setw(8) << "Train" << " | "
                  << std::setw(8) << "Predict" << std::endl;
        std::cout << std::string(45, '-') << std::endl;

        printResults("SVM", svm_accuracy, training_time, prediction_time);
        printResults("Neural Network", nn_accuracy, training_time, prediction_time);
        printResults("K-Means", kmeans_accuracy, training_time, prediction_time);

        // Cleanup
        CUDA_CHECK(cudaFree(predictions));

        // Train ensemble classifier
        std::cout << "\nTraining classifiers..." << std::endl;
        ensemble.train(train_data);

        // Test ensemble classifier
        start = std::chrono::high_resolution_clock::now();
        float accuracy = ensemble.getAccuracy(test_data.features, test_data.labels, test_data.n_samples);
        end = std::chrono::high_resolution_clock::now();
        float ensemble_training_time = std::chrono::duration<float, std::milli>(end - start).count();
        float ensemble_prediction_time = 0.0f; // Assuming prediction time is not provided in the original code

        // Print results
        std::cout << "\nResults:" << std::endl;
        std::cout << std::setw(15) << "Classifier" << " | "
                  << std::setw(10) << "Accuracy" << " | "
                  << std::setw(8) << "Train" << " | "
                  << std::setw(8) << "Predict" << std::endl;
        std::cout << std::string(45, '-') << std::endl;

        printResults("Ensemble", accuracy, ensemble_training_time, ensemble_prediction_time);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
