#include <gtest/gtest.h>
#include "data/iris_data_loader.h"
#include "preprocessing/data_preprocessor.h"
#include "classifiers/svm_classifier.h"
#include "classifiers/neural_network.h"
#include "classifiers/kmeans_classifier.h"
#include "ensemble/ensemble_classifier.h"

class ClassifierTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load test data
        loader.loadData("../data/iris.csv");
        data = loader.getData();
        
        // Preprocess data
        preprocessor.normalizeFeatures(data);
        preprocessor.splitData(data, train_data, test_data, 0.2f);
    }

    IrisDataLoader loader;
    DataPreprocessor preprocessor;
    IrisData data, train_data, test_data;
};

// Test SVM Classifier
TEST_F(ClassifierTest, SVMClassifierTest) {
    SVMClassifier svm;
    svm.train(train_data);
    float accuracy = svm.getAccuracy(test_data.features, 
                                   test_data.labels, 
                                   test_data.n_samples);
    EXPECT_GT(accuracy, 0.8f);  // Expect accuracy > 80%
}

// Test Neural Network
TEST_F(ClassifierTest, NeuralNetworkTest) {
    NeuralNetwork nn(4, 8, 3);  // 4 inputs, 8 hidden, 3 outputs
    nn.train(train_data, 100);  // 100 epochs
    float accuracy = nn.getAccuracy(test_data.features, 
                                  test_data.labels, 
                                  test_data.n_samples);
    EXPECT_GT(accuracy, 0.8f);
}

// Test K-means Classifier
TEST_F(ClassifierTest, KMeansClassifierTest) {
    KMeansClassifier kmeans(3);
    kmeans.train(train_data);
    float accuracy = kmeans.getAccuracy(test_data.features, 
                                      test_data.labels, 
                                      test_data.n_samples);
    EXPECT_GT(accuracy, 0.7f);  // K-means might have lower accuracy
}

// Test Ensemble Classifier
TEST_F(ClassifierTest, EnsembleClassifierTest) {
    EnsembleClassifier ensemble;
    ensemble.train(train_data);
    float accuracy = ensemble.getAccuracy(test_data.features, 
                                        test_data.labels, 
                                        test_data.n_samples);
    EXPECT_GT(accuracy, 0.85f);  // Expect ensemble to perform better
}
