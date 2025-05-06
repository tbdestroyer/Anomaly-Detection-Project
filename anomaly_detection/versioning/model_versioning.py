import json
import os
import shutil
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelVersioning:
    def __init__(self):
        self.models_dir = 'models'
        self.versions_dir = 'models/versions'
        self.backups_dir = 'models/backups'
        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(self.backups_dir, exist_ok=True)
        
    def create_version(self, model_name, model_path, metrics, metadata=None):
        """Create a new model version"""
        # Generate version number
        version = self._get_next_version(model_name)
        
        # Create version directory
        version_dir = f"{self.versions_dir}/{model_name}_v{version}"
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model files
        shutil.copy2(model_path, f"{version_dir}/model.pkl")
        
        # Save version metadata
        version_data = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        with open(f"{version_dir}/metadata.json", 'w') as f:
            json.dump(version_data, f)
            
        # Update version history
        self._update_version_history(model_name, version_data)
        
        return version
        
    def create_backup(self, model_name, model_path):
        """Create a backup of the current model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{self.backups_dir}/{model_name}_{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)
        
        shutil.copy2(model_path, f"{backup_dir}/model.pkl")
        
        return backup_dir
        
    def get_version_history(self, model_name):
        """Get version history for a model"""
        try:
            with open(f"{self.versions_dir}/{model_name}_history.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
            
    def compare_versions(self, model_name, version1, version2):
        """Compare two versions of a model"""
        history = self.get_version_history(model_name)
        versions = {v['version']: v for v in history}
        
        if version1 not in versions or version2 not in versions:
            raise ValueError("One or both versions not found")
            
        comparison = {
            'version1': versions[version1],
            'version2': versions[version2],
            'metrics_comparison': {}
        }
        
        # Compare metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            v1_score = versions[version1]['metrics'].get(metric, 0)
            v2_score = versions[version2]['metrics'].get(metric, 0)
            comparison['metrics_comparison'][metric] = {
                'version1': v1_score,
                'version2': v2_score,
                'difference': v2_score - v1_score
            }
            
        return comparison
        
    def visualize_version_history(self, model_name):
        """Create visualizations for version history"""
        history = self.get_version_history(model_name)
        
        if not history:
            print(f"No version history found for {model_name}")
            return
            
        # Create version timeline
        plt.figure(figsize=(12, 6))
        timestamps = [datetime.fromisoformat(v['timestamp']) for v in history]
        versions = [v['version'] for v in history]
        
        plt.scatter(timestamps, versions)
        plt.title(f'Version Timeline - {model_name}')
        plt.xlabel('Time')
        plt.ylabel('Version')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.versions_dir}/{model_name}_timeline.png")
        plt.close()
        
        # Create metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        plt.figure(figsize=(12, 6))
        
        for metric in metrics:
            values = [v['metrics'].get(metric, 0) for v in history]
            plt.plot(versions, values, label=metric)
            
        plt.title(f'Metrics Comparison - {model_name}')
        plt.xlabel('Version')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.versions_dir}/{model_name}_metrics.png")
        plt.close()
        
    def _get_next_version(self, model_name):
        """Get the next version number for a model"""
        history = self.get_version_history(model_name)
        if not history:
            return 1
        return max(v['version'] for v in history) + 1
        
    def _update_version_history(self, model_name, version_data):
        """Update version history file"""
        history = self.get_version_history(model_name)
        history.append(version_data)
        
        with open(f"{self.versions_dir}/{model_name}_history.json", 'w') as f:
            json.dump(history, f)
            
    def setup_ab_testing(self, model_name, version1, version2, traffic_split=0.5):
        """Set up A/B testing between two versions"""
        ab_config = {
            'model_name': model_name,
            'version1': version1,
            'version2': version2,
            'traffic_split': traffic_split,
            'start_time': datetime.now().isoformat(),
            'metrics': {
                'version1': {'requests': 0, 'successes': 0},
                'version2': {'requests': 0, 'successes': 0}
            }
        }
        
        with open(f"{self.versions_dir}/{model_name}_ab_test.json", 'w') as f:
            json.dump(ab_config, f)
            
        return ab_config
        
    def update_ab_metrics(self, model_name, version, success):
        """Update A/B testing metrics"""
        try:
            with open(f"{self.versions_dir}/{model_name}_ab_test.json", 'r') as f:
                ab_config = json.load(f)
                
            ab_config['metrics'][f'version{version}']['requests'] += 1
            if success:
                ab_config['metrics'][f'version{version}']['successes'] += 1
                
            with open(f"{self.versions_dir}/{model_name}_ab_test.json", 'w') as f:
                json.dump(ab_config, f)
                
        except FileNotFoundError:
            print("A/B testing not configured for this model")
            
    def visualize_ab_results(self, model_name):
        """Create visualizations for A/B testing results"""
        try:
            with open(f"{self.versions_dir}/{model_name}_ab_test.json", 'r') as f:
                ab_config = json.load(f)
                
            # Calculate success rates
            v1_requests = ab_config['metrics']['version1']['requests']
            v1_successes = ab_config['metrics']['version1']['successes']
            v2_requests = ab_config['metrics']['version2']['requests']
            v2_successes = ab_config['metrics']['version2']['successes']
            
            v1_rate = v1_successes / v1_requests if v1_requests > 0 else 0
            v2_rate = v2_successes / v2_requests if v2_requests > 0 else 0
            
            # Create A/B testing results plot
            plt.figure(figsize=(10, 6))
            plt.bar(['Version 1', 'Version 2'], [v1_rate, v2_rate])
            plt.title('A/B Testing Results')
            plt.xlabel('Version')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f"{self.versions_dir}/{model_name}_ab_results.png")
            plt.close()
            
        except FileNotFoundError:
            print("A/B testing not configured for this model") 