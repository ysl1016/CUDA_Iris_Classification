#include "utils/performance_monitor.h"
#include <fstream>
#include <iostream>

std::chrono::high_resolution_clock::time_point PerformanceMonitor::start_time;

void PerformanceMonitor::startTimer() {
    start_time = std::chrono::high_resolution_clock::now();
}

float PerformanceMonitor::stopTimer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                   (end_time - start_time);
    return duration.count() / 1000.0f;  // Convert to seconds
}

size_t PerformanceMonitor::getCurrentMemoryUsage() {
    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    return total_byte - free_byte;
}

void PerformanceMonitor::logMetrics(const char* classifier_name, 
                                  const PerformanceMetrics& metrics) {
    // Create results directory if it doesn't exist
    std::system("mkdir -p results");
    
    std::ofstream log_file("results/performance_log.txt", std::ios::app);
    log_file << "Classifier: " << classifier_name << "\n"
             << "Accuracy: " << metrics.accuracy << "\n"
             << "Precision: " << metrics.precision << "\n"
             << "Recall: " << metrics.recall << "\n"
             << "F1 Score: " << metrics.f1_score << "\n"
             << "Training Time: " << metrics.training_time << "s\n"
             << "Inference Time: " << metrics.inference_time << "s\n"
             << "Memory Usage: " << metrics.memory_usage / 1024.0f / 1024.0f << " MB\n"
             << "----------------------------------------\n";
}
