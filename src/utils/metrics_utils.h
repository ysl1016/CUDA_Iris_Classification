#pragma once
#include "common.h"

namespace MetricsUtils {
    // Calculate classification performance metrics
    float calculateAccuracy(const int* predictions, const int* labels, int n_samples);
    void calculateConfusionMatrix(const int* predictions, 
                                const int* labels, 
                                int n_samples, 
                                int n_classes, 
                                int* confusion_matrix);
    float calculatePrecision(const int* confusion_matrix, 
                           int class_idx, 
                           int n_classes);
    float calculateRecall(const int* confusion_matrix, 
                         int class_idx, 
                         int n_classes);
    float calculateF1Score(float precision, float recall);
}
