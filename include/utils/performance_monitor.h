#pragma once
#include "common.h"

class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor();

    void startTimer();
    void stopTimer();
    float getElapsedTime();
    size_t getCurrentMemoryUsage();
    void recordMetrics(PerformanceMetrics& metrics);

private:
    cudaEvent_t start, stop;
    float elapsed_time;
};
