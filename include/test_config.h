#pragma once

#define COLAB_GPU_MEMORY_LIMIT (15UL * 1024 * 1024 * 1024)  // 15GB
#define COLAB_MAX_BATCH_SIZE 1024
#define COLAB_CUDA_ARCH 75  // Tesla T4

#define TEST_DATA_PATH "data/iris.csv"
#define TEST_RESULTS_PATH "results/"
