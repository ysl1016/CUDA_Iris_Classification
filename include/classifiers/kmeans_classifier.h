#pragma once
#include "common.h"

class KMeansClassifier {
public:
    KMeansClassifier(int n_clusters);
    ~KMeansClassifier();

    void train(const IrisData& data);
    void predict(const float* features, int n_samples, int* predictions);
    float accuracy(const int* predictions, const int* labels, int n_samples);

private:
    int n_clusters;
    int max_iterations;     // maximum iterations for convergence
    float convergence_threshold;  // threshold for centroid movement

    float* d_centroids;     // cluster centroids
    int* d_cluster_labels;  // cluster assignments
    int* d_cluster_sizes;   // size of each cluster
    float* d_new_centroids; // temporary storage for centroid updates
    
    void initializeCentroids(const float* features, int n_samples);
    bool updateCentroids(const float* features, int n_samples);
    void assignClusters(const float* features, int n_samples);
    void mapClustersToClasses(const int* labels, int n_samples);
    
    // Mapping from cluster ID to class label
    int* d_cluster_to_class_map;
};
