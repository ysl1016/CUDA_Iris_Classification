#pragma once
#include "common.h"

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);
    ~NeuralNetwork();

    void train(const IrisData& data, int epochs);
    void predict(const float* features, int* predictions, int n_samples);
    float getAccuracy(const float* features, const int* labels, int n_samples);

private:
    // Network parameters
    float *d_W1, *d_W2;  // Weights
    float *d_b1, *d_b2;  // Biases
    float *d_h;          // Hidden layer activations
    float *d_output;     // Output layer activations
    
    int input_size;
    int hidden_size;
    int output_size;
    
    float* d_weights;
    float learning_rate = 0.01f;
    
    void initializeParameters();
    void forwardPass(const float* features, int n_samples);
    void backwardPass(const float* features, const int* labels, int n_samples);
};
