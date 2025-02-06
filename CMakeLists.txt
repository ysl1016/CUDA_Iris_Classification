cmake_minimum_required(VERSION 3.18)
project(iris_classification CUDA CXX)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
file(GLOB_RECURSE SOURCES 
    "src/*.cu"
    "src/*.cpp"
)

# Create executable
add_executable(iris_classifier ${SOURCES})
set_target_properties(iris_classifier PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create shared library
add_library(iris_classifier_lib SHARED ${SOURCES})
set_target_properties(iris_classifier_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional GTest support
find_package(GTest)
if(GTEST_FOUND)
    enable_testing()
    include_directories(${GTEST_INCLUDE_DIRS})
    
    file(GLOB_RECURSE TEST_SOURCES 
        "tests/*.cu"
        "tests/*.cpp"
    )
    
    add_executable(run_tests ${TEST_SOURCES})
    target_link_libraries(run_tests
        ${GTEST_LIBRARIES}
        pthread
    )
    
    add_test(NAME unit_tests COMMAND run_tests)
endif()
