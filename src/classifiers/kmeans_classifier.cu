#include "kmeans_classifier.h"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <curand_kernel.h>
#include <cfloat>

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
        float min_distance = FLT_MAX;
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

void KMeansClassifier::train(const IrisData& data) {
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_centroids, k * data.n_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cluster_labels, data.n_samples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_sizes, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_centroids, k * data.n_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cluster_to_class_map, k * sizeof(int)));
    
    // Initialize centroids
    initializeCentroids(data.features, data.n_samples);
    
    // Main k-means loop
    bool converged = false;
    int iteration = 0;
    
    while (!converged && iteration < max_iterations) {
        // Assign points to clusters
        assignClusters(data.features, data.n_samples);
        
        // Update centroids
        converged = updateCentroids(data.features, data.n_samples);
        
        iteration++;
    }
    
    // Map clusters to classes
    mapClustersToClasses(data.labels, data.n_samples);
}

void KMeansClassifier::predict(const float* features, int* predictions, int n_samples) {
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
        k
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

float KMeansClassifier::getAccuracy(const float* features, 
                                   const int* labels, 
                                   int n_samples) {
    int* d_predictions;
    CUDA_CHECK(cudaMalloc(&d_predictions, n_samples * sizeof(int)));
    
    predict(features, d_predictions, n_samples);
    
    // Count correct predictions
    thrust::device_ptr<const int> d_labels_ptr(labels);
    thrust::device_ptr<const int> d_pred_ptr(d_predictions);
    
    int correct = thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(d_labels_ptr, d_pred_ptr)),
        thrust::make_zip_iterator(thrust::make_tuple(d_labels_ptr + n_samples, d_pred_ptr + n_samples)),
        [] __device__ (const thrust::tuple<const int&, const int&>& t) {
            return thrust::get<0>(t) == thrust::get<1>(t) ? 1 : 0;
        },
        0,
        thrust::plus<int>()
    );
    
    CUDA_CHECK(cudaFree(d_predictions));
    
    return static_cast<float>(correct) / n_samples;
}

KMeansClassifier::~KMeansClassifier() {
    if (d_centroids) CUDA_CHECK(cudaFree(d_centroids));
    if (d_cluster_labels) CUDA_CHECK(cudaFree(d_cluster_labels));
    if (d_cluster_sizes) CUDA_CHECK(cudaFree(d_cluster_sizes));
    if (d_new_centroids) CUDA_CHECK(cudaFree(d_new_centroids));
    if (d_cluster_to_class_map) CUDA_CHECK(cudaFree(d_cluster_to_class_map));
}
