<<<<<<< HEAD
# CUDA_Iris_Classification
CUDA-based Iris Classification System with multiple classifiers
=======
# Iris Classification System with CUDA GPU Acceleration

A high-performance hybrid classification system for the Iris dataset using CUDA GPU acceleration. This project implements multiple classification algorithms (SVM, Neural Network, K-means) and combines them using an ensemble approach.

## Table of Contents
- [Project Overview](#project-overview)
- [System Requirements](#system-requirements)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

### Features
- Multiple GPU-accelerated classifiers:
  - Support Vector Machine (SVM)
  - Neural Network
  - K-means Clustering
- Ensemble learning system
- CUDA-optimized implementations
- Comprehensive performance metrics
- Real-time visualization capabilities

### Key Components
1. **Data Processing**
   - Efficient data loading and preprocessing
   - Feature normalization
   - Train-test splitting

2. **Classification Algorithms**
   - SVM with RBF kernel
   - Multi-layer Neural Network
   - K-means Clustering
   - Weighted Ensemble System

3. **Performance Optimization**
   - CUDA memory management
   - Parallel computation
   - Optimized kernel configurations

## System Requirements

### Hardware Requirements
- NVIDIA GPU with Compute Capability 6.0 or higher
- Minimum 4GB GPU memory
- 8GB system RAM recommended

### Software Requirements
- CUDA Toolkit 11.0 or higher
- CMake 3.10 or higher
- C++14 compatible compiler
- Python 3.6+ (for visualization)
- Ubuntu 18.04+ / CentOS 7+ / Windows 10

### Dependencies
```bash
# CUDA Dependencies
- CUDA Toolkit
- cuBLAS
- cuRAND

# C++ Dependencies
- Thrust library (included with CUDA)

# Python Dependencies
- numpy
- pandas
- matplotlib
- seaborn
```

## Directory Structure
```
iris_classification_system/
├── bin/                    # Compiled executables
├── lib/                    # External libraries
├── src/
│   ├── data/              # Data loading and management
│   ├── preprocessing/      # Data preprocessing
│   ├── classifiers/       # Classification algorithms
│   ├── ensemble/          # Ensemble system
│   └── utils/             # Utility functions
├── include/               # Header files
├── build/                 # Build directory
├── CMakeLists.txt
├── install.sh
└── run.sh
```

## Installation

### Quick Install
```bash
# Clone the repository
git clone https://github.com/yourusername/iris_classification_system.git
cd iris_classification_system

# Make scripts executable
chmod +x install.sh run.sh

# Install dependencies and build
./install.sh
```

### Manual Installation
```bash
# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make -j4

# Return to project root
cd ..
```

## Usage

### Running the System
```bash
# Using the run script
./run.sh

# Manual execution
./bin/iris_classifier [path_to_data]
```

### Command Line Arguments
```bash
Options:
  --data-path     Path to Iris dataset (default: data/iris.csv)
  --test-ratio    Test set ratio (default: 0.2)
  --epochs        Training epochs for neural network (default: 100)
  --batch-size    Batch size for training (default: 32)
```

### Example Usage
```bash
# Run with default settings
./bin/iris_classifier

# Run with custom parameters
./bin/iris_classifier --data-path custom_data.csv --test-ratio 0.3
```

## Implementation Details

### Data Processing
- Feature normalization using CUDA parallel processing
- Efficient memory transfers between CPU and GPU
- Real-time data augmentation capabilities

### Classification Algorithms

#### SVM Classifier
- RBF kernel implementation
- SMO optimization algorithm
- Parallel kernel computation

#### Neural Network
- Multi-layer perceptron architecture
- ReLU activation functions
- Parallel backpropagation

#### K-means Clustering
- K-means++ initialization
- Parallel centroid updates
- Efficient distance calculations

### Ensemble System
- Weighted voting mechanism
- Dynamic weight adjustment
- Parallel prediction aggregation

## Performance Metrics

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Performance Monitoring
- Training time
- Prediction latency
- GPU memory usage
- Computational efficiency

## Contributing

### Development Guidelines
1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate tests
4. Submit a pull request

### Code Style
- Follow C++ best practices
- Use CUDA coding conventions
- Include appropriate documentation
- Add unit tests for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository for the Iris dataset
- NVIDIA for CUDA toolkit and documentation
- Open source community for various tools and libraries

## Contact

For questions and support, please open an issue in the GitHub repository or contact [eyeon3800@gmail.com].

---

**Note**: This project is part of a research initiative on GPU-accelerated machine learning algorithms. For academic use, please cite appropriately.
>>>>>>> 99cb9da (Initial commit: CUDA-based Iris Classification System)
