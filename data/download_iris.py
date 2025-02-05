import urllib.request
import os

def download_iris_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    save_path = "../data/iris.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download the file
    urllib.request.urlretrieve(url, save_path)
    print(f"Downloaded Iris dataset to {save_path}")

if __name__ == "__main__":
    download_iris_dataset()
