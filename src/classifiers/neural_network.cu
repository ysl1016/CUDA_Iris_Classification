#include "classifiers/neural_network.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

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

void NeuralNetwork::forwardPass(const float* input, int n_samples) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples * output_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    forwardPassKernel<<<grid_size, block_size>>>(
        input,
        d_weights,
        d_bias,
        d_output,
        n_samples,
        input_size,
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
    // Allocate memory for weights and biases
    size_t w1_size = input_size * hidden_size * sizeof(float);
    size_t w2_size = hidden_size * output_size * sizeof(float);
    size_t b1_size = hidden_size * sizeof(float);
    size_t b2_size = output_size * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_W1, w1_size));
    CUDA_CHECK(cudaMalloc(&d_W2, w2_size));
    CUDA_CHECK(cudaMalloc(&d_b1, b1_size));
    CUDA_CHECK(cudaMalloc(&d_b2, b2_size));
    
    // Initialize with random values
    thrust::device_ptr<float> w1_ptr(d_W1);
    thrust::device_ptr<float> w2_ptr(d_W2);
    thrust::device_ptr<float> b1_ptr(d_b1);
    thrust::device_ptr<float> b2_ptr(d_b2);
    
    thrust::fill(thrust::device, w1_ptr, w1_ptr + input_size * hidden_size, 0.01f);
    thrust::fill(thrust::device, w2_ptr, w2_ptr + hidden_size * output_size, 0.01f);
    thrust::fill(thrust::device, b1_ptr, b1_ptr + hidden_size, 0.0f);
    thrust::fill(thrust::device, b2_ptr, b2_ptr + output_size, 0.0f);
}

class NeuralNetwork {
private:
    float* d_weights;
    float* d_bias;
    float learning_rate;
    
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        // Initialize network parameters
        initializeParameters();
    }
    
    ~NeuralNetwork() {
        // Free device memory
        cudaFree(d_W1);
        cudaFree(d_W2);
        cudaFree(d_b1);
        cudaFree(d_b2);
        cudaFree(d_h);
        cudaFree(d_output);
    }

    float getAccuracy(const float* features, const int* labels, int n_samples) {
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
            [=] __device__ (int idx) {
                return d_pred_ptr[idx] == d_labels_ptr[idx] ? 1 : 0;
            },
            0,
            thrust::plus<int>()
        );
        
        CUDA_CHECK(cudaFree(predictions));
        return static_cast<float>(correct) / n_samples;
    }
};
