#pragma once
#include "common.h"

class DataPreprocessor {
public:
    static void standardizeFeatures(IrisData& data);
    static void splitData(const IrisData& data, IrisData& train, IrisData& test, float train_ratio);
    
private:
    static void calculateMeanAndStd(const float* data, int n_samples, int n_features,
                                  float* mean, float* std);
};
