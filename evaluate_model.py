import pandas as pd
import numpy as np
import joblib
import time
import psutil
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import argparse

def log_computation_metrics(env_name, inference_time, throughput, cpu_usage, memory_usage):
    csv_path = 'benchmark_results.csv'
    new_data = pd.DataFrame([{
        'Environment': env_name,
        'Inference Time (s)': inference_time,
        'Throughput (rows/sec)': throughput,
        'CPU Usage (%)': cpu_usage,
        'Memory Usage (MB)': memory_usage
    }])
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if env_name in df['Environment'].values:
            # Update each column individually to avoid shape issues
            for col in new_data.columns:
                df.loc[df['Environment'] == env_name, col] = new_data[col].values[0]
        else:
            df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
        
    df.to_csv(csv_path, index=False)
    print(f"✅ Computation metrics logged for {env_name} in '{csv_path}'")


def evaluate_model(model_path, test_path, output_dir='outputs', n_repeats=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading model and test data...")
    model = joblib.load(model_path)
    test_data = pd.read_csv(test_path)
    
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']

    print(f"Starting evaluation... (Simulating load with {n_repeats} repeats)")

    process = psutil.Process()

    mem_before = process.memory_info().rss / (1024 * 1024)
    cpu_times_before = process.cpu_times()

    # Simulate sustained load
    start_time = time.perf_counter()
    for _ in range(n_repeats):
        y_pred = model.predict(X_test)
    elapsed_time = time.perf_counter() - start_time

    cpu_times_after = process.cpu_times()
    mem_after = process.memory_info().rss / (1024 * 1024)

    # Calculate CPU usage %
    cpu_time_used = (cpu_times_after.user - cpu_times_before.user) + (cpu_times_after.system - cpu_times_before.system)
    cores = psutil.cpu_count(logical=True)
    cpu_usage = (cpu_time_used / elapsed_time) * 100 / cores


    # Calculate throughput
    total_predictions = len(X_test) * n_repeats
    throughput = total_predictions / elapsed_time

    # Adjust predictions (from last run)
    y_pred = np.where(y_pred == 1, 0, 1)
    decision_scores = -model.score_samples(X_test)

    # Model performance (for PDF reporting only)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, decision_scores)
    avg_precision = average_precision_score(y_test, decision_scores)

    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Model Evaluation Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Precision: {precision:.4f}", ln=True)
    pdf.cell(0, 10, f"Recall: {recall:.4f}", ln=True)
    pdf.cell(0, 10, f"F1 Score: {f1:.4f}", ln=True)
    pdf.cell(0, 10, f"AUC-ROC: {auc:.4f}", ln=True)
    pdf.cell(0, 10, f"Avg Precision: {avg_precision:.4f}", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, f"Total Inference Time: {elapsed_time:.4f} s", ln=True)
    pdf.cell(0, 10, f"Throughput: {throughput:.0f} rows/sec", ln=True)
    pdf.cell(0, 10, f"CPU Usage: {cpu_usage:.2f} %", ln=True)
    pdf.cell(0, 10, f"Memory Usage Increase: {mem_after - mem_before:.2f} MB", ln=True)

    pdf_path = os.path.join(output_dir, 'model_performance_report.pdf')
    pdf.output(pdf_path)

    print(f"PDF report saved at {pdf_path}")

    # Return computation metrics for logging
    return elapsed_time, throughput, cpu_usage, mem_after - mem_before


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='Environment name (e.g., Local, Docker (Cloud))')
    args = parser.parse_args()

    elapsed_time, throughput, cpu_usage, memory_usage = evaluate_model('isolation_forest_model.joblib', 'creditcard_test.csv')

    log_computation_metrics(
        env_name=args.env,
        inference_time=elapsed_time,
        throughput=throughput,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage
    )

