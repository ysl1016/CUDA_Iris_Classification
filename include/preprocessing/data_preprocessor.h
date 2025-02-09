#pragma once
#include "common.h"
#include <random>

class DataPreprocessor {
private:
    struct VarianceOp {
        float mean;
        __host__ __device__
        VarianceOp(float m) : mean(m) {}
        __host__ __device__
        float operator()(float x) const {
            float diff = x - mean;
            return diff * diff;
        }
    };

public:
    // Basic preprocessing methods
    static void normalizeFeatures(IrisData& data);
    static void standardizeFeatures(IrisData& data);
    static void splitData(const IrisData& data, IrisData& train, IrisData& test, float train_ratio);
    
    // Advanced preprocessing methods
    void shuffleData(IrisData& data);
    void augmentData(IrisData& data, float noise_std = 0.1f);
    
    static void calculateMeanAndStd(const float* data, int n_samples, int n_features, float* mean, float* std);
    
private:
    // Helper methods
    void addGaussianNoise(float* features, int n_samples, 
                         int n_features, float std_dev);
};
