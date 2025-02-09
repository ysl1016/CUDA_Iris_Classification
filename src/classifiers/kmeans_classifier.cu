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
__global__ void updateCentroidsKernel(
    const float* features,
    const int* cluster_labels,
    float* new_centroids,
    int* cluster_sizes,
    int n_samples,
    int n_features,
    int n_clusters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;
    
    int cluster = cluster_labels[idx];
    atomicAdd(&cluster_sizes[cluster], 1);
    
    for (int j = 0; j < n_features; j++) {
        float feature_val = features[idx * n_features + j];
        atomicAdd(&new_centroids[cluster * n_features + j], feature_val);
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
    CUDA_CHECK(cudaMalloc(&d_centroids, n_clusters * N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cluster_sizes, n_clusters * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_centroids, n_clusters * N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cluster_to_class_map, n_clusters * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_labels, MAX_SAMPLES * sizeof(int)));
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
    float* d_distances = nullptr;
    int* d_nearest_centroids = nullptr;
    
    try {
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
            N_FEATURES,
            n_clusters
        );
        
        // Map cluster assignments to class predictions
        thrust::device_ptr<int> d_nearest_ptr(d_nearest_centroids);
        thrust::device_ptr<int> d_pred_ptr(predictions);
        thrust::device_ptr<int> d_map_ptr(d_cluster_to_class_map);
        
        auto transform_op = [] __device__ (int cluster, thrust::device_ptr<int> map_ptr) {
            return map_ptr[cluster];
        };

        thrust::transform(
            d_nearest_ptr,
            d_nearest_ptr + n_samples,
            d_pred_ptr,
            [=] __device__ (int cluster) { return transform_op(cluster, d_map_ptr); }
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    catch (const std::runtime_error& e) {
        if (d_distances) cudaFree(d_distances);
        if (d_nearest_centroids) cudaFree(d_nearest_centroids);
        throw;
    }
    
    cudaFree(d_distances);
    cudaFree(d_nearest_centroids);
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

KMeansClassifier::~KMeansClassifier() noexcept {
    if (d_centroids) cudaFree(d_centroids);
    if (d_cluster_sizes) cudaFree(d_cluster_sizes);
    if (d_new_centroids) cudaFree(d_new_centroids);
    if (d_cluster_to_class_map) cudaFree(d_cluster_to_class_map);
    if (d_cluster_labels) cudaFree(d_cluster_labels);
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
    // Reset new centroids and cluster sizes
    CUDA_CHECK(cudaMemset(d_new_centroids, 0, n_clusters * N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_cluster_sizes, 0, n_clusters * sizeof(int)));
    
    // Calculate sum of points in each cluster
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    updateCentroidsKernel<<<grid_size, block_size>>>(
        features,
        d_cluster_labels,
        d_new_centroids,
        d_cluster_sizes,
        n_samples,
        N_FEATURES,
        n_clusters
    );
    
    // Check for convergence
    float max_movement = 0.0f;
    thrust::device_ptr<float> d_old_ptr(d_centroids);
    thrust::device_ptr<float> d_new_ptr(d_new_centroids);
    thrust::device_ptr<int> d_sizes_ptr(d_cluster_sizes);
    
    for (int i = 0; i < n_clusters; i++) {
        // Skip empty clusters
        if (d_sizes_ptr[i] == 0) continue;
        
        // Calculate new centroid positions
        for (int j = 0; j < N_FEATURES; j++) {
            int idx = i * N_FEATURES + j;
            float old_pos = d_old_ptr[idx];
            float new_pos = d_new_ptr[idx] / d_sizes_ptr[i];
            d_old_ptr[idx] = new_pos;  // Update centroid position
            
            // Track maximum movement
            float movement = fabs(new_pos - old_pos);
            max_movement = max(max_movement, movement);
        }
    }
    
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

void KMeansClassifier::mapClustersToClasses(const int* labels, int n_samples) {
    // Allocate temporary memory for voting
    int* h_votes = new int[n_clusters * N_CLASSES]();
    int* h_cluster_to_class = new int[n_clusters];
    
    try {
        // Count votes for each cluster
        thrust::device_ptr<const int> d_cluster_labels_ptr(d_cluster_labels);
        thrust::device_ptr<const int> d_labels_ptr(labels);
        
        for (int i = 0; i < n_samples; i++) {
            int cluster = d_cluster_labels_ptr[i];
            int label = d_labels_ptr[i];
            h_votes[cluster * N_CLASSES + label]++;
        }
        
        // Assign each cluster to the majority class
        for (int i = 0; i < n_clusters; i++) {
            int max_votes = -1;
            int majority_class = 0;
            
            for (int j = 0; j < N_CLASSES; j++) {
                int votes = h_votes[i * N_CLASSES + j];
                if (votes > max_votes) {
                    max_votes = votes;
                    majority_class = j;
                }
            }
            
            h_cluster_to_class[i] = majority_class;
        }
        
        // Copy mapping to device
        CUDA_CHECK(cudaMemcpy(d_cluster_to_class_map, h_cluster_to_class, 
                            n_clusters * sizeof(int), cudaMemcpyHostToDevice));
    }
    catch (const std::runtime_error& e) {
        delete[] h_votes;
        delete[] h_cluster_to_class;
        throw;
    }
    
    delete[] h_votes;
    delete[] h_cluster_to_class;
}
