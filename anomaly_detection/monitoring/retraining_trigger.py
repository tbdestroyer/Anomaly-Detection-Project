import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class RetrainingTrigger:
    def __init__(self):
        self.output_dir = 'outputs/monitoring'
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_history = self._load_metrics_history()
        
    def check_retraining_conditions(self, model_name, current_metrics):
        """Check if retraining conditions are met"""
        conditions = {
            'performance_degradation': self._check_performance_degradation(model_name, current_metrics),
            'data_drift': self._check_data_drift(),
            'time_based': self._check_time_based_retraining(model_name)
        }
        
        # Save retraining conditions
        with open(f'{self.output_dir}/retraining_conditions.json', 'w') as f:
            json.dump(conditions, f)
            
        return conditions
        
    def _check_performance_degradation(self, model_name, current_metrics):
        """Check for performance degradation"""
        model_metrics = self.metrics_history[self.metrics_history['model'] == model_name]
        
        if len(model_metrics) < 2:
            return False
            
        # Calculate moving average of metrics
        window_size = min(5, len(model_metrics))
        moving_avg = model_metrics[['accuracy', 'precision', 'recall', 'f1']].rolling(window=window_size).mean().iloc[-1]
        
        # Check for significant degradation
        degradation_threshold = 0.05  # 5% degradation threshold
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if current_metrics[metric] < moving_avg[metric] * (1 - degradation_threshold):
                return True
                
        return False
        
    def _check_data_drift(self):
        """Check for data drift"""
        try:
            with open(f'{self.output_dir}/drift_results.json', 'r') as f:
                drift_results = json.load(f)
                
            # Check if any drift was detected
            for col, test in drift_results['ks_tests'].items():
                if test['drift_detected']:
                    return True
                    
            for col, test in drift_results['chi2_tests'].items():
                if test['drift_detected']:
                    return True
                    
        except FileNotFoundError:
            return False
            
        return False
        
    def _check_time_based_retraining(self, model_name):
        """Check if it's time for scheduled retraining"""
        try:
            with open(f'{self.output_dir}/model_metadata.json', 'r') as f:
                metadata = json.load(f)
                
            if model_name not in metadata:
                return False
                
            last_training = datetime.fromisoformat(metadata[model_name]['last_training'])
            retraining_interval = timedelta(days=metadata[model_name].get('retraining_interval', 30))
            
            return datetime.now() - last_training > retraining_interval
            
        except FileNotFoundError:
            return False
            
    def _load_metrics_history(self):
        """Load historical metrics from CSV"""
        try:
            return pd.read_csv(f'{self.output_dir}/metrics_history.csv')
        except FileNotFoundError:
            return pd.DataFrame(columns=['timestamp', 'model', 'accuracy', 'precision', 'recall', 'f1'])
            
    def update_model_metadata(self, model_name, retraining_interval=30):
        """Update model metadata with last training time"""
        try:
            with open(f'{self.output_dir}/model_metadata.json', 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}
            
        metadata[model_name] = {
            'last_training': datetime.now().isoformat(),
            'retraining_interval': retraining_interval
        }
        
        with open(f'{self.output_dir}/model_metadata.json', 'w') as f:
            json.dump(metadata, f)
            
    def visualize_retraining_conditions(self):
        """Create visualizations for retraining conditions"""
        try:
            with open(f'{self.output_dir}/retraining_conditions.json', 'r') as f:
                conditions = json.load(f)
                
            # Create retraining conditions plot
            plt.figure(figsize=(10, 6))
            conditions_df = pd.DataFrame([conditions])
            conditions_df.plot(kind='bar')
            plt.title('Retraining Conditions')
            plt.xlabel('Condition')
            plt.ylabel('Triggered')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/retraining_conditions.png')
            plt.close()
            
        except FileNotFoundError:
            print("No retraining conditions found") 