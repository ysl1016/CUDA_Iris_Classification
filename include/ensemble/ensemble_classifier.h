#pragma once
#include "common.h"
#include "classifiers/svm_classifier.h"
#include "classifiers/neural_network.h"
#include "classifiers/kmeans_classifier.h"

class EnsembleClassifier {
public:
    EnsembleClassifier();
    
    bool init() {
        if (d_weights == nullptr || d_predictions == nullptr) {
            cleanup();
            return false;
        }
        return true;
    }

    ~EnsembleClassifier();

    void train(const IrisData& data);
    void predict(const float* features, int* predictions, int n_samples);
    float getAccuracy(const float* features, const int* labels, int n_samples);

private:
    SVMClassifier svm;
    NeuralNetwork nn;
    KMeansClassifier kmeans;
    
    float* d_weights;        // Classifier weights
    int* d_predictions;      // Individual classifier predictions
    static const int n_classifiers = 3;
    static const int MAX_SAMPLES = 150;  // Iris dataset size
    
    void updateWeights(const float* features, const int* labels, int n_samples);
    void cleanup() {
        if (d_weights) cudaFree(d_weights);
        if (d_predictions) cudaFree(d_predictions);
        d_weights = nullptr;
        d_predictions = nullptr;
    }
};
