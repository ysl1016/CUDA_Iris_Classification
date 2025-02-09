#pragma once
#include "common.h"

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    ~NeuralNetwork();
    
    void train(const float* features, const int* labels, int n_samples, int epochs = MAX_EPOCHS);
    void predict(const float* features, int* predictions, int n_samples);
    float getAccuracy(const float* features, const int* labels, int n_samples);

private:
    void initializeParameters();
    void forwardPass(const float* features, int n_samples);
    
    int input_size;
    int hidden_size;
    int output_size;
    
    float *d_W1, *d_W2;
    float *d_b1, *d_b2;
    float *d_h, *d_output;
};
