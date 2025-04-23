import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import ClearML tools for logging and experiment tracking
from clearml import Task, Logger

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
    model = joblib.load(model_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and target
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']
    
    # Get predictions and decision scores
    y_pred = model.predict(X_test)
    # Convert predictions to match original labels (0 for normal, 1 for fraud)
    y_pred = np.where(y_pred == 1, 0, 1)
    
    # Get decision scores (negative of the anomaly score)
    # Higher score means more likely to be normal (not fraud)
    decision_scores = -model.score_samples(X_test)
    
    # Calculate metrics
    print("\n=== Model Evaluation Metrics ===")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, decision_scores):.4f}")
    print(f"Average Precision Score: {average_precision_score(y_test, decision_scores):.4f}")
    
    # Log metrics to ClearML for visualization and comparison
    # Log metrics to ClearML for visualization and comparison
    logger.report_scalar("Metrics", "Precision", int(0), precision_score(y_test, y_pred))
    logger.report_scalar("Metrics", "Recall", int(0), recall_score(y_test, y_pred))
    logger.report_scalar("Metrics", "F1 Score", int(0), f1_score(y_test, y_pred))
    logger.report_scalar("Metrics", "AUC-ROC", int(0), roc_auc_score(y_test, decision_scores))
    logger.report_scalar("Metrics", "Average Precision", int(0), average_precision_score(y_test, decision_scores))


        
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
    model_path = 'isolation_forest_model.joblib'
    test_path = 'creditcard_test.csv'
    performance = evaluate_model(model_path, test_path) 