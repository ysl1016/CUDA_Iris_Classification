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
    
    backwardPassKernel<<<grid_size, block_size>>>(
        d_weights,
        d_output,
        input,
        labels,
        learning_rate,
        n_samples,
        input_size,
        output_size
    );
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

void NeuralNetwork::initializeParameters() {
    // Allocations
    size_t w1_size = input_size * hidden_size * sizeof(float);
    size_t w2_size = hidden_size * output_size * sizeof(float);
    size_t b1_size = hidden_size * sizeof(float);
    size_t b2_size = output_size * sizeof(float);
    size_t h_size = hidden_size * sizeof(float);
    size_t output_size_bytes = output_size * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_W1, w1_size));
    CUDA_CHECK(cudaMalloc(&d_W2, w2_size));
    CUDA_CHECK(cudaMalloc(&d_b1, b1_size));
    CUDA_CHECK(cudaMalloc(&d_b2, b2_size));
    CUDA_CHECK(cudaMalloc(&d_h, h_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size_bytes));
    
    // Xavier initialization
    float w1_scale = sqrt(2.0f / input_size);
    float w2_scale = sqrt(2.0f / hidden_size);
    
    thrust::device_ptr<float> w1_ptr(d_W1);
    thrust::device_ptr<float> w2_ptr(d_W2);
    thrust::device_ptr<float> b1_ptr(d_b1);
    thrust::device_ptr<float> b2_ptr(d_b2);
    
    // CUDA 난수 생성을 위한 커널
    auto init_weights = [] __device__ (float scale, unsigned int seed) {
        curandState state;
        curand_init(seed, threadIdx.x, 0, &state);
        return scale * (2.0f * curand_uniform(&state) - 1.0f);
    };
    
    unsigned int seed = static_cast<unsigned int>(time(nullptr));
    
    thrust::transform(thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(input_size * hidden_size),
        w1_ptr,
        [=] __device__ (int idx) { return init_weights(w1_scale, seed + idx); }
    );
    
    thrust::transform(thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(hidden_size * output_size),
        w2_ptr,
        [=] __device__ (int idx) { return init_weights(w2_scale, seed + idx); }
    );
    
    thrust::fill(thrust::device, b1_ptr, b1_ptr + hidden_size, 0.0f);
    thrust::fill(thrust::device, b2_ptr, b2_ptr + output_size, 0.0f);
}

float NeuralNetwork::getAccuracy(const float* features, const int* labels, int n_samples) {
    int* predictions;
    CUDA_CHECK(cudaMalloc(&predictions, n_samples * sizeof(int)));
    
    predict(features, predictions, n_samples);
    
    // Calculate accuracy using thrust
    thrust::device_ptr<const int> d_pred_ptr(predictions);
    thrust::device_ptr<const int> d_labels_ptr(labels);
    
    int correct = thrust::transform_reduce(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n_samples),
        [=] __device__ (int idx) -> int {
            return d_pred_ptr[idx] == d_labels_ptr[idx] ? 1 : 0;
        },
        0,
        thrust::plus<int>()
    );
    
    CUDA_CHECK(cudaFree(predictions));
    return static_cast<float>(correct) / n_samples;
}
