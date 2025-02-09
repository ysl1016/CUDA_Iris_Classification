#include "preprocessing/data_preprocessor.h"
#include "utils/metrics_utils.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <random>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

__global__ void computeMeanKernel(const float* features, 
                                 float* mean, 
                                 int n_samples, 
                                 int n_features) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feature_idx < n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            sum += features[i * n_features + feature_idx];
        }
        mean[feature_idx] = sum / n_samples;
    }
}

__global__ void computeStdKernel(const float* features, 
                                const float* mean,
                                float* std,
                                int n_samples,
                                int n_features) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (feature_idx < n_features) {
        float sum_sq = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float diff = features[i * n_features + feature_idx] - mean[feature_idx];
            sum_sq += diff * diff;
        }
        std[feature_idx] = sqrt(sum_sq / n_samples);
    }
}

__global__ void normalizeKernel(float* features, const float* min_vals, 
                               const float* max_vals, int n_samples, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n_samples * n_features; i += stride) {
        int feature_idx = i % n_features;
        float min_val = min_vals[feature_idx];
        float max_val = max_vals[feature_idx];
        if (max_val > min_val) {
            features[i] = (features[i] - min_val) / (max_val - min_val);
        }
    }
}

__global__ void standardizeKernel(float* features, const float* mean, const float* std,
                                int n_samples, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples * n_features) {
        int feature_idx = idx % n_features;
        features[idx] = (features[idx] - mean[feature_idx]) / std[feature_idx];
    }
}

void DataPreprocessor::normalizeFeatures(IrisData& data) {
    thrust::device_vector<float> d_min(data.n_features);
    thrust::device_vector<float> d_max(data.n_features);
    
    // Calculate min and max values for each feature
    for (int f = 0; f < data.n_features; ++f) {
        thrust::device_ptr<float> d_feature = thrust::device_pointer_cast(data.features + f);
        thrust::device_ptr<float> d_end = d_feature + data.n_samples * data.n_features;
        d_min[f] = *thrust::min_element(thrust::device, d_feature, d_end);
        d_max[f] = *thrust::max_element(thrust::device, d_feature, d_end);
    }
    
    // Launch normalization kernel
    int block_size = BLOCK_SIZE;
    int num_blocks = (data.n_samples * data.n_features + block_size - 1) / block_size;
    
    normalizeKernel<<<num_blocks, block_size>>>(
        data.features,
        thrust::raw_pointer_cast(d_min.data()),
        thrust::raw_pointer_cast(d_max.data()),
        data.n_samples,
        data.n_features
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DataPreprocessor::calculateMeanAndStd(const float* data, int n_samples, int n_features,
                                         float* mean, float* std) {
    // Compute mean using kernel
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_features + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    computeMeanKernel<<<grid_size, block_size>>>(
        data, mean, n_samples, n_features
    );
    
    // Compute standard deviation using kernel
    computeStdKernel<<<grid_size, block_size>>>(
        data, mean, std, n_samples, n_features
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DataPreprocessor::standardizeFeatures(IrisData& data) {
    float* d_mean;
    float* d_std;
    CUDA_CHECK(cudaMalloc(&d_mean, data.n_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_std, data.n_features * sizeof(float)));
    
    // Calculate mean and std
    calculateMeanAndStd(data.features, data.n_samples, data.n_features, d_mean, d_std);
    
    // Launch standardization kernel
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((data.n_samples * data.n_features + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    standardizeKernel<<<grid_size, block_size>>>(
        data.features,
        d_mean,
        d_std,
        data.n_samples,
        data.n_features
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_std));
}

void DataPreprocessor::shuffleData(IrisData& data) {
    thrust::device_vector<int> indices(data.n_samples);
    thrust::sequence(indices.begin(), indices.end());
    
    // Create random number generator
    thrust::default_random_engine rng(std::random_device{}());
    
    // Shuffle indices
    thrust::shuffle(indices.begin(), indices.end(), rng);
    
    // Create temporary storage for shuffled data
    thrust::device_vector<float> temp_features(data.n_samples * data.n_features);
    thrust::device_vector<int> temp_labels(data.n_samples);
    
    // Shuffle features and labels according to indices
    for (int i = 0; i < data.n_samples; ++i) {
        int idx = indices[i];
        thrust::copy_n(thrust::device_pointer_cast(data.features + idx * data.n_features),
                      data.n_features,
                      temp_features.begin() + i * data.n_features);
        temp_labels[i] = data.labels[idx];
    }
    
    // Copy back to original arrays
    thrust::copy(temp_features.begin(), temp_features.end(),
                thrust::device_pointer_cast(data.features));
    thrust::copy(temp_labels.begin(), temp_labels.end(),
                thrust::device_pointer_cast(data.labels));
}

void DataPreprocessor::augmentData(IrisData& data, float noise_std) {
    thrust::default_random_engine rng(std::random_device{}());
    thrust::normal_distribution<float> dist(0.0f, noise_std);
    
    int total_elements = data.n_samples * data.n_features;
    thrust::device_vector<float> noise(total_elements);
    
    // Generate noise using device-side RNG
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(total_elements),
        noise.begin(),
        [rng, dist] __device__ (int idx) mutable {
            rng.discard(idx);
            return dist(rng);
        }
    );
    
    // Add noise to features
    thrust::transform(thrust::device_pointer_cast(data.features),
                     thrust::device_pointer_cast(data.features + total_elements),
                     noise.begin(),
                     thrust::device_pointer_cast(data.features),
                     thrust::plus<float>());
}

void DataPreprocessor::splitData(const IrisData& data, IrisData& train, IrisData& test, float train_ratio) {
    int n_train = static_cast<int>(data.n_samples * train_ratio);
    int n_test = data.n_samples - n_train;
    
    train.n_features = data.n_features;
    test.n_features = data.n_features;
    train.n_classes = data.n_classes;
    test.n_classes = data.n_classes;
    
    // Allocate memory for train and test sets
    CUDA_CHECK(cudaMalloc(&train.features, n_train * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&train.labels, n_train * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&test.features, n_test * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&test.labels, n_test * sizeof(int)));
    
    // Copy data
    CUDA_CHECK(cudaMemcpy(train.features, data.features, n_train * 4 * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(train.labels, data.labels, n_train * sizeof(int), 
                         cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(test.features, data.features + n_train * 4, 
                         n_test * 4 * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(test.labels, data.labels + n_train, 
                         n_test * sizeof(int), cudaMemcpyDeviceToDevice));
    
    train.n_samples = n_train;
    test.n_samples = n_test;
}
