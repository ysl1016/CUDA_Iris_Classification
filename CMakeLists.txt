cmake_minimum_required(VERSION 3.10)
project(iris_classification CUDA CXX)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA NVCC flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --extended-lambda")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/data
)

# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cu"
    "src/*.cpp"
)

# Create executable
cuda_add_executable(iris_classifier ${SOURCES})
set_target_properties(iris_classifier PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

# Create shared library
cuda_add_library(iris_classifier_lib SHARED ${SOURCES})
set_target_properties(iris_classifier_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)

# Link libraries
target_link_libraries(iris_classifier
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
)

target_link_libraries(iris_classifier_lib
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
)

# Enable testing
enable_testing()

# Find GTest package
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add test executable
file(GLOB_RECURSE TEST_SOURCES 
    "tests/*.cu"
    "tests/*.cpp"
)

cuda_add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests
    ${GTEST_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    pthread
)

# Add test
add_test(NAME unit_tests COMMAND run_tests)
