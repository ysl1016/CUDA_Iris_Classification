#include "classifiers/neural_network.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

__global__ void forwardPassKernel(const float* input,
                                 const float* weights,
                                 const float* bias,
                                 float* output,
                                 int n_samples,
                                 int input_size,
                                 int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_samples * output_size) {
        int sample = idx / output_size;
        int neuron = idx % output_size;
        
        float sum = bias[neuron];
        for (int i = 0; i < input_size; i++) {
            sum += input[sample * input_size + i] * weights[i * output_size + neuron];
        }
        
        // ReLU activation
        output[idx] = fmaxf(0.0f, sum);
    }
}

__global__ void backwardPassKernel(float* d_weights,
                                 const float* d_output,
                                 const float* d_input,
                                 const int* labels,
                                 float learning_rate,
                                 int n_samples,
                                 int input_size,
                                 int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_size * output_size) {
        int input_idx = idx / output_size;
        int output_idx = idx % output_size;
        
        float gradient = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float error = d_output[i * output_size + output_idx] - (labels[i] == output_idx ? 1.0f : 0.0f);
            gradient += error * d_input[i * input_size + input_idx];
        }
        
        d_weights[idx] -= learning_rate * gradient / n_samples;
    }
}

void NeuralNetwork::train(const IrisData& data, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        forwardPass(data.features, data.n_samples);
        backwardPass(data.features, data.labels, data.n_samples);
    }
}

void NeuralNetwork::predict(const float* features, int* predictions, int n_samples) {
    forwardPass(features, n_samples);
    
    // Get predictions from output layer
    thrust::device_ptr<float> d_output_ptr(d_output);
    thrust::device_ptr<int> d_pred_ptr(predictions);
    
    for (int i = 0; i < n_samples; i++) {
        thrust::device_ptr<float> row_begin = d_output_ptr + i * output_size;
        thrust::device_ptr<float> row_end = row_begin + output_size;
        d_pred_ptr[i] = thrust::max_element(row_begin, row_end) - row_begin;
    }
}
