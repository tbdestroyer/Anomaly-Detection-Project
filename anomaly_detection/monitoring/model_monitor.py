import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelMonitor:
    def __init__(self):
        self.output_dir = 'outputs/monitoring'
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_history = self._load_metrics_history()
        
    def evaluate_model(self, model_name, y_true, y_pred, y_prob=None):
        """Evaluate model performance and track metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
            
        # Save predictions for confusion matrix
        predictions = {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        }
        with open(f'{self.output_dir}/{model_name}_predictions.json', 'w') as f:
            json.dump(predictions, f)
            
        # Update metrics history
        self._update_metrics_history(metrics)
        
        # Check for performance degradation
        self._check_performance_degradation(model_name, metrics)
        
        return metrics
        
    def visualize_performance(self, model_name):
        """Create visualizations for model performance"""
        model_metrics = self.metrics_history[self.metrics_history['model'] == model_name]
        
        if len(model_metrics) == 0:
            print(f"No metrics found for model {model_name}")
            return
            
        # Create performance trend plot
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            plt.plot(model_metrics['timestamp'], model_metrics[metric], label=metric)
        plt.title(f'Performance Metrics Over Time - {model_name}')
        plt.xlabel('Time')
        plt.ylabel('Score')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{model_name}_performance.png')
        plt.close()
        
        # Create confusion matrix
        try:
            with open(f'{self.output_dir}/{model_name}_predictions.json', 'r') as f:
                predictions = json.load(f)
                cm = confusion_matrix(predictions['y_true'], predictions['y_pred'])
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/{model_name}_confusion_matrix.png')
                plt.close()
        except FileNotFoundError:
            print(f"No predictions found for model {model_name}")
            
    def _load_metrics_history(self):
        """Load historical metrics from CSV"""
        try:
            return pd.read_csv(f'{self.output_dir}/metrics_history.csv')
        except FileNotFoundError:
            return pd.DataFrame(columns=['timestamp', 'model', 'accuracy', 'precision', 'recall', 'f1'])
            
    def _update_metrics_history(self, metrics):
        """Update metrics history with new metrics"""
        df = pd.DataFrame([metrics])
        if os.path.exists(f'{self.output_dir}/metrics_history.csv'):
            df.to_csv(f'{self.output_dir}/metrics_history.csv', mode='a', header=False, index=False)
        else:
            df.to_csv(f'{self.output_dir}/metrics_history.csv', index=False)
            
    def _check_performance_degradation(self, model_name, current_metrics):
        """Check for performance degradation and trigger alerts"""
        model_metrics = self.metrics_history[self.metrics_history['model'] == model_name]
        
        if len(model_metrics) < 2:
            return
            
        # Calculate moving average of metrics
        window_size = min(5, len(model_metrics))
        moving_avg = model_metrics[['accuracy', 'precision', 'recall', 'f1']].rolling(window=window_size).mean().iloc[-1]
        
        # Check for significant degradation
        degradation_threshold = 0.05  # 5% degradation threshold
        alerts = []
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if current_metrics[metric] < moving_avg[metric] * (1 - degradation_threshold):
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'model': model_name,
                    'metric': metric,
                    'current_value': current_metrics[metric],
                    'expected_value': moving_avg[metric],
                    'degradation_percentage': (moving_avg[metric] - current_metrics[metric]) / moving_avg[metric] * 100
                })
                
        if alerts:
            # Save alerts
            with open(f'{self.output_dir}/performance_alerts.json', 'w') as f:
                json.dump(alerts, f)
                
            # Print alerts
            for alert in alerts:
                print(f"ALERT: {alert['model']} - {alert['metric']} degraded by {alert['degradation_percentage']:.2f}%") 