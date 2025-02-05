import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def load_results(results_path):
    """Load classification results from JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_confusion_matrix(cm, classes, save_path):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_comparison(results, save_path):
    """Plot accuracy comparison between classifiers"""
    classifiers = list(results['accuracy'].keys())
    accuracies = list(results['accuracy'].values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classifiers, accuracies)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy Comparison')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history, save_path):
    """Plot training history for neural network"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create results directory if it doesn't exist
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)
    
    # Load results
    results = load_results(results_dir / 'classification_results.json')
    
    # Plot confusion matrices
    for classifier, cm in results['confusion_matrices'].items():
        plot_confusion_matrix(
            np.array(cm),
            classes=['setosa', 'versicolor', 'virginica'],
            save_path=results_dir / f'{classifier}_confusion_matrix.png'
        )
    
    # Plot accuracy comparison
    plot_accuracy_comparison(
        results,
        save_path=results_dir / 'accuracy_comparison.png'
    )
    
    # Plot neural network training history
    if 'neural_network_history' in results:
        plot_training_history(
            results['neural_network_history'],
            save_path=results_dir / 'training_history.png'
        )

if __name__ == '__main__':
    main()
