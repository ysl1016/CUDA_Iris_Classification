#include <iostream>
#include <iomanip>
#include <chrono>
#include "data/iris_data_loader.h"
#include "preprocessing/data_preprocessor.h"
#include "ensemble/ensemble_classifier.h"
#include "utils/metrics_utils.h"

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

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
