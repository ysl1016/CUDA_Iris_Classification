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

int main() {
    // data load
    IrisData data;
    // data load code
    
    // preprocessing
    DataPreprocessor::standardizeFeatures(data);
    
    // split data
    IrisData train_data, test_data;
    DataPreprocessor::splitData(data, train_data, test_data, 0.8f);
    
    // create model and train
    NeuralNetwork model(IrisData::n_features, 8, IrisData::n_classes);
    model.train(train_data.features, train_data.labels, train_data.n_samples);
    
    // evaluate
    float accuracy = model.getAccuracy(
        test_data.features, test_data.labels, test_data.n_samples
    );
    printf("Test accuracy: %.2f%%\n", accuracy * 100);
    
    return 0;
}
