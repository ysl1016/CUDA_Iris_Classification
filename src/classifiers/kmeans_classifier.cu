#include "classifiers/kmeans_classifier.h"
#include "utils/metrics_utils.h"
#include <float.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/extrema.h>
#include <curand_kernel.h>
#include <thrust/iterator/discard_iterator.h>

// Initialize centroids using k-means++ algorithm
__global__ void initializeCentroidsKernel(const float* features,
                                         float* centroids,
                                         int n_samples,
                                         int n_features,
                                         unsigned int seed) {
    curandState state;
    curand_init(seed, threadIdx.x, 0, &state);
    
    // Randomly select first centroid
    if (threadIdx.x == 0) {
        int first_idx = curand(&state) % n_samples;
        for (int j = 0; j < n_features; j++) {
            centroids[j] = features[first_idx * n_features + j];
        }
    }
}

// Compute distances to nearest centroids
__global__ void computeDistancesKernel(const float* features,
                                      const float* centroids,
                                      float* distances,
                                      int* nearest_centroids,
                                      int n_samples,
                                      int n_features,
                                      int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_samples) {
        float min_distance = 3.402823466e+38f;  // float max value
        int nearest_centroid = 0;
        
        // Find nearest centroid
        for (int i = 0; i < k; i++) {
            float distance = 0.0f;
            for (int j = 0; j < n_features; j++) {
                float diff = features[idx * n_features + j] - 
                           centroids[i * n_features + j];
                distance += diff * diff;
            }
            
            if (distance < min_distance) {
                min_distance = distance;
                nearest_centroid = i;
            }
        }
        
        distances[idx] = min_distance;
        nearest_centroids[idx] = nearest_centroid;
    }
}

// Update centroids based on mean of assigned points
__global__ void updateCentroidsKernel(const float* features,
                                     const int* cluster_labels,
                                     float* new_centroids,
                                     int* cluster_sizes,
                                     int n_samples,
                                     int n_features,
                                     int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < k * n_features) {
        int cluster = idx / n_features;
        int feature = idx % n_features;
        
        float sum = 0.0f;
        int count = 0;
        
        for (int i = 0; i < n_samples; i++) {
            if (cluster_labels[i] == cluster) {
                sum += features[i * n_features + feature];
                count++;
            }
        }
        
        if (count > 0) {
            new_centroids[cluster * n_features + feature] = sum / count;
            if (feature == 0) {
                cluster_sizes[cluster] = count;
            }
        }
    }
}

// Map clusters to classes based on majority voting
__global__ void mapClustersToClassesKernel(const int* cluster_labels,
                                          const int* true_labels,
                                          int* cluster_to_class_map,
                                          int n_samples,
                                          int k,
                                          int n_classes) {
    int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (cluster < k) {
        int* class_counts = new int[n_classes]();
        
        // Count class occurrences in this cluster
        for (int i = 0; i < n_samples; i++) {
            if (cluster_labels[i] == cluster) {
                class_counts[true_labels[i]]++;
            }
        }
        
        // Find majority class
        int max_count = 0;
        int majority_class = 0;
        for (int i = 0; i < n_classes; i++) {
            if (class_counts[i] > max_count) {
                max_count = class_counts[i];
                majority_class = i;
            }
        }
        
        cluster_to_class_map[cluster] = majority_class;
        delete[] class_counts;
    }
}

// Add this struct before the KMeansClassifier class implementation
struct CompareLabels {
    const thrust::device_ptr<const int> pred;
    const thrust::device_ptr<const int> labels;
    
    __host__ __device__
    CompareLabels(thrust::device_ptr<const int> p, thrust::device_ptr<const int> l) 
        : pred(p), labels(l) {}
    
    __host__ __device__
    int operator()(int idx) const {
        return pred[idx] == labels[idx] ? 1 : 0;
    }
};

KMeansClassifier::KMeansClassifier(int clusters) 
    : n_clusters(clusters), max_iterations(100), convergence_threshold(1e-4) {
    CUDA_CHECK(cudaMalloc(&d_centroids, n_clusters * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cluster_sizes, n_clusters * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_centroids, n_clusters * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cluster_to_class_map, n_clusters * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_labels, n_clusters * sizeof(int)));
}

void KMeansClassifier::train(const IrisData& data) {
    // Initialize centroids
    initializeCentroids(data.features, data.n_samples);

    // Iterate until convergence
    bool converged = false;
    int iteration = 0;
    while (!converged && iteration < max_iterations) {
        assignClusters(data.features, data.n_samples);
        converged = updateCentroids(data.features, data.n_samples);
        iteration++;
    }

    // Map clusters to classes
    mapClustersToClasses(data.labels, data.n_samples);
}

void KMeansClassifier::predict(const float* features, int n_samples, int* predictions) {
    float* d_distances;
    int* d_nearest_centroids;
    
    CUDA_CHECK(cudaMalloc(&d_distances, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nearest_centroids, n_samples * sizeof(int)));
    
    // Compute distances to centroids
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    computeDistancesKernel<<<grid_size, block_size>>>(
        features,
        d_centroids,
        d_distances,
        d_nearest_centroids,
        n_samples,
        4,  // n_features for Iris
        n_clusters
    );
    
    // Map cluster assignments to class predictions
    thrust::device_ptr<int> d_nearest_ptr(d_nearest_centroids);
    thrust::device_ptr<int> d_pred_ptr(predictions);
    thrust::device_ptr<int> d_map_ptr(d_cluster_to_class_map);
    
    thrust::transform(
        d_nearest_ptr,
        d_nearest_ptr + n_samples,
        d_pred_ptr,
        [d_map_ptr] __device__ (int cluster) {
            return d_map_ptr[cluster];
        }
    );
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_nearest_centroids));
}

float KMeansClassifier::accuracy(const int* predictions, const int* labels, int n_samples) {
    thrust::device_ptr<const int> d_labels_ptr(labels);
    thrust::device_ptr<const int> d_pred_ptr(predictions);
    
    CompareLabels compare(d_pred_ptr, d_labels_ptr);
    int correct = thrust::transform_reduce(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n_samples),
        compare,
        0,
        thrust::plus<int>()
    );
    
    return static_cast<float>(correct) / n_samples;
}

KMeansClassifier::~KMeansClassifier() {
    if (d_centroids) CUDA_CHECK(cudaFree(d_centroids));
    if (d_cluster_sizes) CUDA_CHECK(cudaFree(d_cluster_sizes));
    if (d_new_centroids) CUDA_CHECK(cudaFree(d_new_centroids));
    if (d_cluster_to_class_map) CUDA_CHECK(cudaFree(d_cluster_to_class_map));
    if (d_cluster_labels) CUDA_CHECK(cudaFree(d_cluster_labels));
}

void KMeansClassifier::initializeCentroids(const float* features, int n_samples) {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    initializeCentroidsKernel<<<grid_size, block_size>>>(
        features,
        d_centroids,
        n_samples,
        4,  // n_features for Iris
        static_cast<unsigned int>(time(nullptr))
    );
}

void KMeansClassifier::assignClusters(const float* features, int n_samples) {
    float* d_distances;
    CUDA_CHECK(cudaMalloc(&d_distances, n_samples * n_clusters * sizeof(float)));
    
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    computeDistancesKernel<<<grid_size, block_size>>>(
        features,
        d_centroids,
        d_distances,
        d_cluster_labels,
        n_samples,
        4,  // n_features for Iris
        n_clusters
    );
    
    CUDA_CHECK(cudaFree(d_distances));
}

bool KMeansClassifier::updateCentroids(const float* features, int n_samples) {
    thrust::device_ptr<const int> d_assignments(d_cluster_labels);
    thrust::device_ptr<const float> d_features(features);
    thrust::device_ptr<float> d_new_centroids_ptr(d_new_centroids);
    
    // Zero out new centroids
    thrust::fill(d_new_centroids_ptr, d_new_centroids_ptr + n_clusters * 4, 0.0f);
    
    // Create temporary storage for cluster sizes
    thrust::device_vector<int> d_cluster_sizes_vec(n_clusters, 0);
    thrust::device_ptr<int> d_sizes_ptr = d_cluster_sizes_vec.data();
    
    // Copy cluster labels to device vector to ensure they're in device memory
    thrust::device_vector<int> d_labels(d_cluster_labels, d_cluster_labels + n_samples);
    thrust::device_vector<float> d_feature_values(features, features + n_samples * 4);
    
    // Perform reduction by key operation using device vectors
    thrust::device_vector<int> d_keys_output(n_samples);
    thrust::device_vector<float> d_values_output(n_samples);
    
    auto result = thrust::reduce_by_key(
        thrust::device,  // execution policy
        d_labels.begin(),
        d_labels.end(),
        d_feature_values.begin(),
        d_keys_output.begin(),
        d_values_output.begin(),
        thrust::equal_to<int>(),
        thrust::plus<float>()
    );
    
    // Compute means
    for (int c = 0; c < n_clusters; ++c) {
        int size = d_cluster_sizes_vec[c];
        if (size > 0) {
            for (int f = 0; f < 4; ++f) {
                d_new_centroids_ptr[c * 4 + f] = d_values_output[c * 4 + f] / size;
            }
        }
    }
    
    // Calculate maximum centroid movement
    float max_movement = 0.0f;
    for (int i = 0; i < n_clusters * 4; ++i) {
        float diff = abs(d_centroids[i] - d_new_centroids[i]);
        max_movement = max(max_movement, diff);
    }
    
    // Update centroids
    thrust::copy(d_new_centroids_ptr, d_new_centroids_ptr + n_clusters * 4, thrust::device_ptr<float>(d_centroids));
    
    return max_movement < convergence_threshold;
}

float KMeansClassifier::getAccuracy(const float* features, const int* labels, int n_samples) {
    int* predictions;
    CUDA_CHECK(cudaMalloc(&predictions, n_samples * sizeof(int)));
    
    // Get predictions
    predict(features, n_samples, predictions);
    
    // Calculate accuracy
    float acc = accuracy(predictions, labels, n_samples);
    
    CUDA_CHECK(cudaFree(predictions));
    return acc;
}
