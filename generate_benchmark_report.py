import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the benchmark results
benchmark_df = pd.read_csv('benchmark_results.csv')

# Plot 1: Inference Time Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=benchmark_df, x='Environment', y='Inference Time (s)', color='skyblue')
plt.ylabel('Inference Time (s)')
plt.title('Inference Time Across Environments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('inference_time_comparison.png')
plt.close()

# Plot 2: Throughput Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=benchmark_df, x='Environment', y='Throughput (rows/sec)', color='lightgreen')
plt.ylabel('Throughput (rows/sec)')
plt.title('Throughput Across Environments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('throughput_comparison.png')
plt.close()

# Plot 3: Resource Usage
plt.figure(figsize=(10, 6))
resource_data = pd.melt(benchmark_df, 
                       id_vars=['Environment'],
                       value_vars=['CPU Usage (%)', 'Memory Usage (MB)'],
                       var_name='Metric',
                       value_name='Value')
sns.barplot(data=resource_data, x='Environment', y='Value', hue='Metric')
plt.title('Resource Usage Across Environments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('resource_usage_comparison.png')
plt.close()

print("Benchmark visualization charts generated successfully!")
print("\nSummary of Results:")
print(benchmark_df.to_string(index=False))
