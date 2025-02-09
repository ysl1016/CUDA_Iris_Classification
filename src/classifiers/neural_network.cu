#include "classifiers/neural_network.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <curand_kernel.h>
#include <time.h>

// Constants
#define LEARNING_RATE 0.01f
#define BLOCK_SIZE 256

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

__global__ void computeGradientsKernel(
    float* W1, float* W2,
    float* output,
    const float* input,
    const int* labels,
    float learning_rate,
    int n_samples,
    int input_size,
    int hidden_size,
    int output_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples * output_size) return;
    
    int sample_idx = idx / output_size;
    int class_idx = idx % output_size;
    
    // Compute gradient for output layer
    float target = (labels[sample_idx] == class_idx) ? 1.0f : 0.0f;
    float error = output[idx] - target;
    
    // Update weights (simplified backpropagation)
    for (int i = 0; i < hidden_size; i++) {
        int w2_idx = i * output_size + class_idx;
        W2[w2_idx] -= learning_rate * error * input[sample_idx * input_size + i];
    }
}

__global__ void initializeWeightsKernel(float* weights, int size, float scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        weights[idx] = scale * (2.0f * curand_uniform(&state) - 1.0f);
    }
}

void NeuralNetwork::forwardPass(const float* features, int n_samples) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples * output_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // First layer
    forwardPassKernel<<<grid_size, block_size>>>(
        features,
        d_W1,
        d_b1,
        d_h,
        n_samples,
        input_size,
        hidden_size
    );
    
    // Second layer
    forwardPassKernel<<<grid_size, block_size>>>(
        d_h,
        d_W2,
        d_b2,
        d_output,
        n_samples,
        hidden_size,
        output_size
    );
}

void NeuralNetwork::backwardPass(const float* input, const int* labels, int n_samples) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples * output_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    computeGradientsKernel<<<grid_size, block_size>>>(
        d_W1,
        d_W2,
        d_output,
        input,
        labels,
        learning_rate,
        n_samples,
        input_size,
        hidden_size,
        output_size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void NeuralNetwork::train(const float* features, const int* labels, int n_samples, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        forwardPass(features, n_samples);
        backwardPass(features, labels, n_samples);
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

void NeuralNetwork::initializeParameters() {
    // Allocations
    size_t w1_size = input_size * hidden_size;
    size_t w2_size = hidden_size * output_size;
    
    CUDA_CHECK(cudaMalloc(&d_W1, w1_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, w2_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
    
    // Xavier initialization
    float w1_scale = sqrt(2.0f / input_size);
    float w2_scale = sqrt(2.0f / hidden_size);
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size1((w1_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grid_size2((w2_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Initialize weights
    initializeWeightsKernel<<<grid_size1, block_size>>>(d_W1, w1_size, w1_scale, seed);
    initializeWeightsKernel<<<grid_size2, block_size>>>(d_W2, w2_size, w2_scale, seed);
    
    // Initialize biases to zero
    CUDA_CHECK(cudaMemset(d_b1, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b2, 0, output_size * sizeof(float)));
}

float NeuralNetwork::getAccuracy(const float* features, const int* labels, int n_samples) {
    int* predictions;
    CUDA_CHECK(cudaMalloc(&predictions, n_samples * sizeof(int)));
    
    predict(features, predictions, n_samples);
    float accuracy = MetricsUtils::calculateAccuracy(predictions, labels, n_samples);
    CUDA_CHECK(cudaFree(predictions));
    return accuracy;
}

NeuralNetwork::~NeuralNetwork() {
    cleanup();
}

void NeuralNetwork::cleanup() {
    if (d_W1) cudaFree(d_W1);
    if (d_W2) cudaFree(d_W2);
    if (d_b1) cudaFree(d_b1);
    if (d_b2) cudaFree(d_b2);
    if (d_h) cudaFree(d_h);
    if (d_output) cudaFree(d_output);
}
