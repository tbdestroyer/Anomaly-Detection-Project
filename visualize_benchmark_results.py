import pandas as pd
import matplotlib.pyplot as plt
import os

# Define output folder
output_folder = 'benchmark_charts'

# Create folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load benchmark results
df = pd.read_csv('benchmark_results.csv')

# 1️⃣ Inference Time Comparison
plt.figure(figsize=(8,5))
plt.bar(df['Environment'], df['Inference Time (s)'], color='cornflowerblue')
plt.ylabel('Inference Time (s)')
plt.title('Inference Time Across Environments')
plt.savefig(f'{output_folder}/inference_time_comparison.png')
plt.close()

# 2️⃣ Throughput Comparison
plt.figure(figsize=(8,5))
plt.bar(df['Environment'], df['Throughput (rows/sec)'], color='mediumseagreen')
plt.ylabel('Throughput (rows/sec)')
plt.title('Throughput Across Environments')
plt.savefig(f'{output_folder}/throughput_comparison.png')
plt.close()

# 3️⃣ CPU Usage Change
plt.figure(figsize=(8,5))
plt.bar(df['Environment'], df['CPU Usage Change (%)'], color='salmon')
plt.ylabel('CPU Usage Change (%)')
plt.title('CPU Usage Variation Across Environments')
plt.savefig(f'{output_folder}/cpu_usage_comparison.png')
plt.close()

# 4️⃣ Memory Usage Increase
plt.figure(figsize=(8,5))
plt.bar(df['Environment'], df['Memory Usage Increase (MB)'], color='gold')
plt.ylabel('Memory Usage Increase (MB)')
plt.title('Memory Usage Across Environments')
plt.savefig(f'{output_folder}/memory_usage_comparison.png')
plt.close()

print(f"All charts saved in the '{output_folder}' folder!")
