import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
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
    new_data.to_csv(csv_path, index=False)
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

    # Calculate metrics
    print("\n=== Model Evaluation Metrics ===")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, decision_scores):.4f}")
    print(f"Average Precision Score: {average_precision_score(y_test, decision_scores):.4f}")
    
    # Create ROC curve
    plt.figure(figsize=(12, 5))
    
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
    
    # Return computation metrics for logging
    return elapsed_time, throughput, cpu_usage, mem_after - mem_before, performance_report

if __name__ == "__main__":
    elapsed_time, throughput, cpu_usage, memory_usage, performance_report = evaluate_model(
        'outputs/isolation_forest_model.joblib', 
        'api_simulation_data.csv'
    )
    
    log_computation_metrics(
        env_name='Local',
        inference_time=elapsed_time,
        throughput=throughput,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage
    )

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
        # Create output directory
        self.output_dir = Path('outputs/metrics')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluation results
        self.evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'predictions': {}
        }
    
    def evaluate(self):
        """Perform comprehensive model evaluation."""
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_scores = self.model.predict_proba(self.X_test)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.decision_function(self.X_test)
        
        # Calculate metrics
        self.evaluation_results['metrics'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred)
        }
        
        # Store predictions
        self.evaluation_results['predictions'] = {
            'y_true': self.y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_scores': y_scores.tolist()
        }
        
        # Save evaluation results
        self._save_evaluation_results()
        
        # Log metrics to CSV
        self._log_metrics_to_csv()
        
        return self.evaluation_results
    
    def _save_evaluation_results(self):
        """Save evaluation results to JSON files."""
        # Save metrics
        metrics_file = self.output_dir / f'{self.model.__class__.__name__.lower()}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.evaluation_results['metrics'], f, indent=4)
        
        # Save predictions
        predictions_file = self.output_dir / f'{self.model.__class__.__name__.lower()}_predictions.json'
        with open(predictions_file, 'w') as f:
            json.dump(self.evaluation_results['predictions'], f, indent=4)
    
    def _log_metrics_to_csv(self):
        """Log metrics to CSV file for tracking over time."""
        csv_file = self.output_dir / 'metrics_log.csv'
        
        # Create metrics row
        metrics_row = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model.__class__.__name__,
            **self.evaluation_results['metrics']
        }
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics_row])
        
        # Append to existing CSV or create new one
        if csv_file.exists():
            metrics_df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(csv_file, index=False)
    
    def visualize_results(self):
        """Generate visualizations of evaluation results."""
        y_true = self.evaluation_results['predictions']['y_true']
        y_pred = self.evaluation_results['predictions']['y_pred']
        y_scores = self.evaluation_results['predictions']['y_scores']
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Confusion Matrix
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Plot 2: ROC Curve
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Plot 3: Precision-Recall Curve
        plt.subplot(1, 3, 3)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / f'{self.model.__class__.__name__.lower()}_evaluation.png'
        plt.savefig(output_file)
        plt.close()
    
    def get_evaluation_history(self):
        """Retrieve historical evaluation results."""
        metrics_file = self.output_dir / f'{self.model.__class__.__name__.lower()}_metrics.json'
        predictions_file = self.output_dir / f'{self.model.__class__.__name__.lower()}_predictions.json'
        
        history = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                history['metrics'] = json.load(f)
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                history['predictions'] = json.load(f)
        
        return history
    
    def compare_models(self, other_model_evaluator):
        """Compare this model's performance with another model."""
        # Get metrics for both models
        this_metrics = self.evaluation_results['metrics']
        other_metrics = other_model_evaluator.evaluation_results['metrics']
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            self.model.__class__.__name__: this_metrics,
            other_model_evaluator.model.__class__.__name__: other_metrics
        })
        
        # Calculate differences
        comparison['Difference'] = comparison[self.model.__class__.__name__] - comparison[other_model_evaluator.model.__class__.__name__]
        
        # Save comparison
        comparison_file = self.output_dir / 'model_comparison.csv'
        comparison.to_csv(comparison_file)
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        comparison[comparison.columns[:2]].plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / 'model_comparison.png'
        plt.savefig(output_file)
        plt.close()
        
        return comparison
