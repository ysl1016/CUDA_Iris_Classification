#pragma once
#include "common.h"
#include <chrono>

class PerformanceMonitor {
public:
    static void startTimer();
    static float stopTimer();
    static size_t getCurrentMemoryUsage();
    static void logMetrics(const char* classifier_name, 
                          const PerformanceMetrics& metrics);
private:
    static std::chrono::high_resolution_clock::time_point start_time;
};
