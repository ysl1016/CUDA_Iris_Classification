#include "data_preprocessor.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

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

__global__ void standardizeKernel(float* features, const float* mean, 
                                 const float* std, int n_samples, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n_samples * n_features; i += stride) {
        int feature_idx = i % n_features;
        if (std[feature_idx] > 0) {
            features[i] = (features[i] - mean[feature_idx]) / std[feature_idx];
        }
    }
}

void DataPreprocessor::normalizeFeatures(IrisData& data) {
    thrust::device_vector<float> d_min(data.n_features);
    thrust::device_vector<float> d_max(data.n_features);
    
    // Calculate min and max values for each feature
    for (int f = 0; f < data.n_features; ++f) {
        thrust::device_ptr<float> d_feature = thrust::device_pointer_cast(data.features + f);
        d_min[f] = *thrust::min_element(d_feature, d_feature + data.n_samples * data.n_features, data.n_features);
        d_max[f] = *thrust::max_element(d_feature, d_feature + data.n_samples * data.n_features, data.n_features);
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

void DataPreprocessor::standardizeFeatures(IrisData& data) {
    thrust::device_vector<float> d_mean(data.n_features);
    thrust::device_vector<float> d_std(data.n_features);
    
    // Calculate mean and standard deviation
    calculateMeanAndStd(data.features, data.n_samples, 
                       data.n_features, 
                       thrust::raw_pointer_cast(d_mean.data()),
                       thrust::raw_pointer_cast(d_std.data()));
    
    // Launch standardization kernel
    int block_size = BLOCK_SIZE;
    int num_blocks = (data.n_samples * data.n_features + block_size - 1) / block_size;
    
    standardizeKernel<<<num_blocks, block_size>>>(
        data.features,
        thrust::raw_pointer_cast(d_mean.data()),
        thrust::raw_pointer_cast(d_std.data()),
        data.n_samples,
        data.n_features
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
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
    // Add Gaussian noise to features
    thrust::default_random_engine rng(std::random_device{}());
    thrust::normal_distribution<float> dist(0.0f, noise_std);
    
    int total_elements = data.n_samples * data.n_features;
    thrust::device_vector<float> noise(total_elements);
    
    // Generate noise
    thrust::transform(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(total_elements),
                     noise.begin(),
                     [=] __device__ (int idx) {
                         thrust::default_random_engine rng_local = rng;
                         rng_local.discard(idx);
                         return dist(rng_local);
                     });
    
    // Add noise to features
    thrust::transform(thrust::device_pointer_cast(data.features),
                     thrust::device_pointer_cast(data.features + total_elements),
                     noise.begin(),
                     thrust::device_pointer_cast(data.features),
                     thrust::plus<float>());
}
