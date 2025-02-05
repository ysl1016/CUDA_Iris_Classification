#pragma once
#include "common.h"
#include "classifiers/svm_classifier.h"
#include "classifiers/neural_network.h"
#include "classifiers/kmeans_classifier.h"

class EnsembleClassifier {
public:
    EnsembleClassifier() : nn(4, 8, 3), kmeans(3) {
        CUDA_CHECK(cudaMalloc(&d_weights, n_classifiers * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_predictions, n_classifiers * sizeof(int)));
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
    
    void updateWeights(const float* features, const int* labels, int n_samples);
};
