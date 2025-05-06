from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import confusion_matrix
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

from clearml import Task, Logger
from fpdf import FPDF
import os
import argparse
from PIL import Image


def evaluate_model(model_path, test_path, output_dir='outputs', n_repeats=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading model and test data...")
    if model_path.endswith(".h5"):
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)
    else:
        model = joblib.load(model_path)

    model_name = os.path.basename(model_path).replace("_model.joblib", "").replace(".joblib", "").capitalize()

    task = Task.init(
        project_name="Anomaly Detection",
        task_name=f"Evaluate {model_name} ({os.path.basename(test_path)})",
        task_type="testing",
    )

    print(f"🔗 ClearML Task URL: {task.get_output_log_web_page()}")

    logger = task.get_logger()

    test_data = pd.read_csv(test_path)
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']
    if y_test.ndim > 1:
        y_test = y_test.ravel()

    print(f"Starting evaluation... (Simulating load with {n_repeats} repeats)")

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)
    cpu_times_before = process.cpu_times()

    start_time = time.perf_counter()
    for _ in range(n_repeats):
        if hasattr(model, "predict") and not hasattr(model, "score_samples"):
            model.predict(X_test)
        elif hasattr(model, "score_samples"):
            model.score_samples(X_test)
        else:
            raise ValueError("Unsupported model type.")
    elapsed_time = time.perf_counter() - start_time

    cpu_times_after = process.cpu_times()
    mem_after = process.memory_info().rss / (1024 * 1024)

    cpu_time_used = (cpu_times_after.user - cpu_times_before.user) + (cpu_times_after.system - cpu_times_before.system)
    cores = psutil.cpu_count(logical=True)
    cpu_usage = (cpu_time_used / elapsed_time) * 100 / cores

    total_predictions = len(X_test) * n_repeats
    throughput = total_predictions / elapsed_time

    # === Model evaluation logic based on type ===
    if hasattr(model, "predict") and not hasattr(model, "score_samples"):
        # Keras autoencoder
        X_pred = model.predict(X_test)
        reconstruction_error = np.mean(np.square(X_test - X_pred), axis=1)
        threshold_path = model_path.replace("_model.h5", "_threshold.joblib")
        threshold = joblib.load(threshold_path)
        y_pred = (reconstruction_error > threshold).astype(int)
        decision_scores = -reconstruction_error

    elif hasattr(model, "score_samples"):
        # Sklearn anomaly detection
        decision_scores = model.score_samples(X_test)
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)

    else:
        raise ValueError("Unsupported model type.")

    # === Metrics ===
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, decision_scores)
    avg_precision = average_precision_score(y_test, decision_scores)

    # === PDF Report ===
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
    print(f"📄 PDF report saved at {pdf_path}")

    # === Confusion Matrix and Curves ===
    logger.report_scalar("Metrics", "Precision", precision, iteration=0)
    logger.report_scalar("Metrics", "Recall", recall, iteration=0)
    logger.report_scalar("Metrics", "F1 Score", f1, iteration=0)
    logger.report_scalar("Metrics", "AUC-ROC", auc, iteration=0)
    logger.report_scalar("Metrics", "Average Precision", avg_precision, iteration=0)

    cm = confusion_matrix(y_test, y_pred)
    logger.report_confusion_matrix("Confusion Matrix", "Model Evaluation", cm.tolist(), iteration=0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, decision_scores)
    plt.plot(fpr, tpr, label=f'AUC-ROC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, decision_scores)
    plt.plot(recall_vals, precision_vals, label=f'AP = {avg_precision:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.tight_layout()
    curve_path = os.path.join(output_dir, 'model_evaluation_curves.png')
    plt.savefig(curve_path)
    img = Image.open(curve_path)
    logger.report_image("Evaluation Curves", "ROC + PR", iteration=0, image=img)

    # === Save CSV Report ===
    performance_report = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'Average Precision'],
        'Value': [precision, recall, f1, auc, avg_precision]
    })
    report_path = os.path.join(output_dir, 'model_performance_report.csv')
    performance_report.to_csv(report_path, index=False)
    print(f"📈 CSV performance report saved at {report_path}")
    print(f"📊 Evaluation curves saved at {curve_path}")
    task.close()
    return elapsed_time, throughput, cpu_usage, mem_after - mem_before, performance_report

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
            for col in new_data.columns:
                df.loc[df['Environment'] == env_name, col] = new_data[col].values[0]
        else:
            df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
        
    df.to_csv(csv_path, index=False)
    print(f"✅ Computation metrics logged for {env_name} in '{csv_path}'")

# 💡 THIS IS THE MISSING PART!
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help='Environment name (e.g., Local, Docker, Zaratan)')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (e.g., outputs/lof_model.joblib)')
    parser.add_argument('--test', type=str, default='data/creditcard_test.csv', help='Path to test CSV file')
    parser.add_argument('--n_repeats', type=int, default=10, help='Number of times to repeat inference')

    args = parser.parse_args()

    elapsed_time, throughput, cpu_usage, memory_usage, performance_report = evaluate_model(
        model_path=args.model,
        test_path=args.test,
        n_repeats=args.n_repeats
    )

    log_computation_metrics(
        env_name=args.env,
        inference_time=elapsed_time,
        throughput=throughput,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage
    )
