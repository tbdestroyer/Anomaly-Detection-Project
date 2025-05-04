import requests
import pandas as pd
import json
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Load data
print("Loading test data...")
data = pd.read_csv('api_simulation_data.csv')
features = data.drop('Class', axis=1)
labels = data['Class']

# Number of samples to test
n_samples = 100  # You can change this number
print(f"Testing with {n_samples} samples...")

# Prepare data
sample_data = features.head(n_samples).values.tolist()
true_labels = labels.head(n_samples).values

# Send request and measure time
start_time = time.time()
response = requests.post(url, json={"data": sample_data})
latency = (time.time() - start_time) * 1000  # Convert to milliseconds

# Process response
if response.status_code == 200:
    result = response.json()
    predictions = result['predictions']
    
    # Calculate metrics
    report = classification_report(true_labels, predictions, output_dict=True)
    
    # Print results
    print("\n📊 Test Results:")
    print(f"Latency: {latency:.2f} ms")
    print(f"Throughput: {n_samples/(latency/1000):.2f} samples/second")
    print("\nClassification Report:")
    print(f"Precision (Fraud): {report['1']['precision']:.4f}")
    print(f"Recall (Fraud): {report['1']['recall']:.4f}")
    print(f"F1-Score (Fraud): {report['1']['f1-score']:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('test_confusion_matrix.png')
    print("\n✅ Confusion matrix saved as 'test_confusion_matrix.png'")
    
else:
    print(f"❌ Failed with status code {response.status_code}")
    print(response.text)
