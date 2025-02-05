#pragma once
#include "common.h"
#include <random>

class DataPreprocessor {
public:
    // Basic preprocessing methods
    void normalizeFeatures(IrisData& data);
    void standardizeFeatures(IrisData& data);
    void splitData(const IrisData& data, IrisData& train_data, 
                  IrisData& test_data, float test_ratio);
    
    // Advanced preprocessing methods
    void shuffleData(IrisData& data);
    void augmentData(IrisData& data, float noise_std = 0.1f);
    
private:
    // Helper methods
    void calculateMeanAndStd(const float* features, int n_samples, 
                            int n_features, float* mean, float* std);
    void addGaussianNoise(float* features, int n_samples, 
                         int n_features, float std_dev);
};
