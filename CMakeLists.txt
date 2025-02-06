cmake_minimum_required(VERSION 3.10)
project(iris_classifier CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add executable
cuda_add_executable(iris_classifier 
    src/main.cu
    src/classifiers/neural_network.cu
    src/classifiers/kmeans_classifier.cu
    src/classifiers/svm_classifier.cu
    src/ensemble/ensemble_classifier.cu
    src/utils/metrics_utils.cu
    src/data/iris_data_loader.cu
    src/preprocessing/data_preprocessor.cu
)

# Link libraries
target_link_libraries(iris_classifier ${CUDA_LIBRARIES} ${GTEST_LIBRARIES} pthread)
