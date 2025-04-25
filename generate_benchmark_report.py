import pandas as pd
import matplotlib.pyplot as plt

# Define your benchmark data
data = {
    'Environment': ['Local', 'Docker (Local)', 'Docker (Cloud)'],
    'Inference Time (s)': [0.140, 0.1633, 0.120],  # Example values
    'Throughput (rows/sec)': [400000, 348822, 475000],
    'F1 Score': [0.1816, 0.1816, 0.1816],
    'CPU Usage Change (%)': [12.5, -10.4, 8.2],
    'Memory Usage Increase (MB)': [2.0, 1.52, 1.8]
}

# Create DataFrame
benchmark_df = pd.DataFrame(data)

# Save to CSV
benchmark_df.to_csv('benchmark_results.csv', index=False)

# Plot 1: Inference Time Comparison
plt.figure(figsize=(8,5))
plt.bar(benchmark_df['Environment'], benchmark_df['Inference Time (s)'], color='skyblue')
plt.ylabel('Inference Time (s)')
plt.title('Inference Time Across Environments')
plt.savefig('inference_time_comparison.png')
plt.close()

# Plot 2: Throughput Comparison
plt.figure(figsize=(8,5))
plt.bar(benchmark_df['Environment'], benchmark_df['Throughput (rows/sec)'], color='lightgreen')
plt.ylabel('Throughput (rows/sec)')
plt.title('Throughput Across Environments')
plt.savefig('throughput_comparison.png')
plt.close()

print("Benchmark CSV and charts generated successfully!")
