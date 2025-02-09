#pragma once
#include "common.h"

class DataPreprocessor {
public:
    static void normalizeFeatures(IrisData& data);
    static void standardizeFeatures(IrisData& data);
    static void shuffleData(IrisData& data);
    static void augmentData(IrisData& data, float noise_std = 0.1f);
    static void splitData(const IrisData& data, IrisData& train, IrisData& test, float train_ratio);
    
private:
    static void calculateMeanAndStd(const float* data, int n_samples, int n_features,
                                  float* mean, float* std);
};
