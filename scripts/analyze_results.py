import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def analyze_performance_log(log_file):
    """Analyze and visualize performance metrics from log file"""
    metrics = []
    current_metrics = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('Classifier:'):
                if current_metrics:
                    metrics.append(current_metrics)
                current_metrics = {}
                current_metrics['Classifier'] = line.split(': ')[1].strip()
            elif ': ' in line:
                key, value = line.split(': ')
                try:
                    current_metrics[key] = float(value.strip().replace('s', '').replace('MB', ''))
                except ValueError:
                    continue
    
    if current_metrics:
        metrics.append(current_metrics)
    
    df = pd.DataFrame(metrics)
    
    # Create results directory if it doesn't exist
    Path('results').mkdir(exist_ok=True)
    
    # Save metrics to CSV
    df.to_csv('results/performance_metrics.csv', index=False)
    
    # Create visualizations
    plt.style.use('seaborn')
    
    # 1. Accuracy and F1 Score Comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='Classifier', y='Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='Classifier', y='F1 Score')
    plt.title('F1 Score Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('results/accuracy_metrics.png')
    plt.close()
    
    # 2. Time and Memory Usage
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='Classifier', y='Training Time')
    plt.title('Training Time (seconds)')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='Classifier', y='Memory Usage')
    plt.title('Memory Usage (MB)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png')
    plt.close()
    
    # 3. Precision-Recall Plot
    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        plt.scatter(row['Precision'], row['Recall'], 
                   label=row['Classifier'], s=100)
    
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/precision_recall.png')
    plt.close()
    
    # Generate summary report
    with open('results/summary_report.txt', 'w') as f:
        f.write("Performance Analysis Summary\n")
        f.write("==========================\n\n")
        
        f.write("Best Accuracy: {:.2f}% ({})\n".format(
            df['Accuracy'].max() * 100,
            df.loc[df['Accuracy'].idxmax(), 'Classifier']
        ))
        
        f.write("Best F1 Score: {:.2f} ({})\n".format(
            df['F1 Score'].max(),
            df.loc[df['F1 Score'].idxmax(), 'Classifier']
        ))
        
        f.write("Fastest Training: {:.3f}s ({})\n".format(
            df['Training Time'].min(),
            df.loc[df['Training Time'].idxmin(), 'Classifier']
        ))
        
        f.write("Most Memory Efficient: {:.1f}MB ({})\n".format(
            df['Memory Usage'].min(),
            df.loc[df['Memory Usage'].idxmin(), 'Classifier']
        ))

def main():
    try:
        analyze_performance_log('results/performance_log.txt')
        print("Analysis completed successfully!")
        print("Results saved in the 'results' directory:")
        print("  - performance_metrics.csv")
        print("  - accuracy_metrics.png")
        print("  - performance_metrics.png")
        print("  - precision_recall.png")
        print("  - summary_report.txt")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == '__main__':
    main()
