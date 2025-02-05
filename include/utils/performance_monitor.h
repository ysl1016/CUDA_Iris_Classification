#pragma once
#include "common.h"
#include <chrono>

class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor();

    void startTimer();
    float stopTimer();
    size_t getCurrentMemoryUsage();
    void logMetrics(const char* classifier_name, const PerformanceMetrics& metrics);

private:
    cudaEvent_t start, stop;
    static std::chrono::high_resolution_clock::time_point start_time;
    float elapsed_time;
};
