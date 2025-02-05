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

        // Train and evaluate SVM
        auto start = std::chrono::high_resolution_clock::now();
        svm.train(train_data);
        auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        float svm_accuracy = svm.getAccuracy(test_data.features, test_data.labels, test_data.n_samples);
        auto pred_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        printResults("SVM", svm_accuracy, train_time, pred_time);

        // Train and evaluate Neural Network
        start = std::chrono::high_resolution_clock::now();
        nn.train(train_data, 100);  // 100 epochs
        train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        float nn_accuracy = nn.getAccuracy(test_data.features, test_data.labels, test_data.n_samples);
        pred_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        printResults("Neural Network", nn_accuracy, train_time, pred_time);

        // Train and evaluate K-means
        start = std::chrono::high_resolution_clock::now();
        kmeans.train(train_data);
        train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        float kmeans_accuracy = kmeans.getAccuracy(test_data.features, test_data.labels, test_data.n_samples);
        pred_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        printResults("K-means", kmeans_accuracy, train_time, pred_time);

        // Train and evaluate Ensemble
        start = std::chrono::high_resolution_clock::now();
        ensemble.train(train_data);
        train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        float ensemble_accuracy = ensemble.getAccuracy(test_data.features, test_data.labels, test_data.n_samples);
        pred_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        printResults("Ensemble", ensemble_accuracy, train_time, pred_time);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
