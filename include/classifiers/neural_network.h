#pragma once
#include "common.h"

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
          learning_rate(LEARNING_RATE) {
        initializeParameters();
    }
    
    ~NeuralNetwork();
    
    void train(const float* features, const int* labels, int n_samples, int epochs = MAX_EPOCHS);
    void predict(const float* features, int* predictions, int n_samples);
    float getAccuracy(const float* features, const int* labels, int n_samples);

private:
    int input_size;
    int hidden_size;
    int output_size;
    float learning_rate;
    
    float* d_W1;
    float* d_W2;
    float* d_b1;
    float* d_b2;
    float* d_h;
    float* d_output;
    
    void initializeParameters();
    void forwardPass(const float* input, int n_samples);
    void backwardPass(const float* input, const int* labels, int n_samples);
    void cleanup();
};
