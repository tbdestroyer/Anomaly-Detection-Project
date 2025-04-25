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
import seaborn as sns

# Import ClearML tools for logging and experiment tracking
from clearml import Task, Logger
from fpdf import FPDF
import os
import argparse
# Initialize a ClearML Task for evaluation tracking
task = Task.init(
    project_name="Anomaly Detection",               # Same project name to keep experiments grouped
    task_name="Evaluate Isolation Forest (Cloud)",  # A meaningful label for this evaluation run
    task_type="testing"                             # Indicates this is a testing/evaluation run
)

# Create a logger instance for reporting custom metrics to the ClearML dashboard
logger = task.get_logger()


def evaluate_model(model_path, test_path):
    # Load the trained model and test data
    print("Loading model and test data...")


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


def evaluate_model(model_path, test_path, output_dir='outputs', n_repeats=100):
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

    
    # Calculate metrics
    print("\n=== Model Evaluation Metrics ===")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, decision_scores):.4f}")
    print(f"Average Precision Score: {average_precision_score(y_test, decision_scores):.4f}")
    
    # Log metrics to ClearML for visualization and comparison
    # Log metrics to ClearML for visualization and comparison
 # Log metrics to ClearML for visualization and comparison
    logger.report_scalar(title="Metrics", series="Precision", value=precision_score(y_test, y_pred), iteration=0)
    logger.report_scalar(title="Metrics", series="Recall", value=recall_score(y_test, y_pred), iteration=0)
    logger.report_scalar(title="Metrics", series="F1 Score", value=f1_score(y_test, y_pred), iteration=0)
    logger.report_scalar(title="Metrics", series="AUC-ROC", value=roc_auc_score(y_test, decision_scores), iteration=0)
    logger.report_scalar(title="Metrics", series="Average Precision", value=average_precision_score(y_test, decision_scores), iteration=0)


        
    # Create ROC curve
    plt.figure(figsize=(12, 5))
    
    # ROC curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, decision_scores)
    plt.plot(fpr, tpr, label=f'AUC-ROC = {roc_auc_score(y_test, decision_scores):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Precision-Recall curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test, decision_scores)
    plt.plot(recall, precision, label=f'AP = {average_precision_score(y_test, decision_scores):.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_evaluation_curves.png')
    
    # Create a detailed performance report
    performance_report = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'Average Precision'],
        'Value': [
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, decision_scores),
            average_precision_score(y_test, decision_scores)
        ]
    })
    
    # Save the performance report
    performance_report.to_csv('model_performance_report.csv', index=False)
    print("\nPerformance report saved as 'model_performance_report.csv'")
    print("Evaluation curves saved as 'model_evaluation_curves.png'")
    
    return performance_report

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

