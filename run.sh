#!/bin/bash

# Set error handling
set -e

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    case $color in
        "green") echo -e "\033[32m$message\033[0m" ;;
        "red") echo -e "\033[31m$message\033[0m" ;;
        "yellow") echo -e "\033[33m$message\033[0m" ;;
        *) echo "$message" ;;
    esac
}

# Function to check CUDA availability
check_cuda() {
    if ! command -v nvcc &>/dev/null; then
        print_status "red" "Error: CUDA toolkit not found!"
        print_status "yellow" "Please ensure CUDA toolkit is installed and in PATH"
        exit 1
    fi
    
    print_status "green" "CUDA toolkit found"
}

# Function to check Python dependencies
check_python_deps() {
    local required_packages="numpy pandas matplotlib seaborn"
    local missing_packages=""
    
    for package in $required_packages; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages="$missing_packages $package"
        fi
    done
    
    if [ ! -z "$missing_packages" ]; then
        print_status "yellow" "Installing missing Python packages:$missing_packages"
        pip3 install $missing_packages
    fi
    
    print_status "green" "All Python dependencies are satisfied"
}

# Clean previous results
clean_results() {
    print_status "yellow" "Cleaning previous results..."
    rm -rf results/*
    mkdir -p results
    print_status "green" "Results directory cleaned"
}

# Set CUDA device (if multiple GPUs available)
export CUDA_VISIBLE_DEVICES=0

# Print header
echo "================================"
echo "Iris Classification System"
echo "================================"

# Check requirements
print_status "yellow" "Checking requirements..."
check_cuda
check_python_deps

# Clean previous results
clean_results

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    print_status "yellow" "Creating build directory..."
    mkdir build
fi

# Build project
print_status "yellow" "Building project..."
cd build
cmake .. && make -j4
if [ $? -ne 0 ]; then
    print_status "red" "Build failed!"
    exit 1
fi
print_status "green" "Build successful"

# Run tests
print_status "yellow" "Running tests..."
./run_tests
if [ $? -ne 0 ]; then
    print_status "red" "Tests failed! Please check the test output above"
    exit 1
fi
print_status "green" "All tests passed"

# Run main program
print_status "yellow" "Running main program..."
./iris_classifier
if [ $? -ne 0 ]; then
    print_status "red" "Program execution failed!"
    exit 1
fi
print_status "green" "Program executed successfully"

# Generate performance analysis
cd ..
print_status "yellow" "Generating performance analysis..."
python3 scripts/analyze_results.py
if [ $? -ne 0 ]; then
    print_status "red" "Performance analysis failed!"
    exit 1
fi

# Print summary
echo
print_status "green" "Execution completed successfully!"
echo "----------------------------------------"
echo "Results have been saved to the results directory:"
echo "  - Performance metrics: results/performance_metrics.csv"
echo "  - Visualizations: results/*.png"
echo "  - Summary report: results/summary_report.txt"
echo "----------------------------------------"

# Optional: Display summary report if it exists
if [ -f "results/summary_report.txt" ]; then
    echo
    echo "Summary Report:"
    echo "----------------------------------------"
    cat results/summary_report.txt
fi
