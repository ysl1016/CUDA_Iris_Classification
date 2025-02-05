#pragma once
#include "common.h"

class SVMClassifier {
public:
    SVMClassifier(float C = 1.0f, float gamma = 0.1f);
    ~SVMClassifier();

    void train(const IrisData& data);
    void predict(const float* features, int* predictions, int n_samples);
    float getAccuracy(const float* features, const int* labels, int n_samples);

private:
    float* d_kernel_matrix;
    float* d_alpha;
    float* d_support_vectors;
    int* d_support_vector_labels;
    int n_support_vectors;
    
    float C;        // regularization parameter
    float gamma;    // RBF kernel parameter
    
    void computeKernelMatrix(const float* features, int n_samples);
    void optimizeDual();
};
