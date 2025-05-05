import pandas as pd
import numpy as np
from scipy import stats
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup logging
logging.basicConfig(
    filename='logs/drift_monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DriftDetector:
    def __init__(self, training_data_path='creditcard_train.csv', features=None):
        self.training_data = pd.read_csv(training_data_path)
        self.features = features or self.training_data.columns.drop(['Class'])
        self.drift_history = []
        
    def detect_drift(self, new_data, threshold=0.05):
        """Detect drift between training and new data"""
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'features': {},
            'overall_drift_score': 0
        }
        
        for feature in self.features:
            # KS test for distribution drift
            ks_stat, p_value = stats.ks_2samp(
                self.training_data[feature], 
                new_data[feature]
            )
            
            # Wasserstein distance for magnitude of drift
            wasserstein_dist = stats.wasserstein_distance(
                self.training_data[feature],
                new_data[feature]
            )
            
            drift_report['features'][feature] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'wasserstein_distance': wasserstein_dist,
                'drift_detected': p_value < threshold
            }
            
            if p_value < threshold:
                logging.warning(f"Drift detected in feature {feature}: p-value={p_value:.4f}")
        
        # Calculate overall drift score
        drift_scores = [report['wasserstein_distance'] for report in drift_report['features'].values()]
        drift_report['overall_drift_score'] = np.mean(drift_scores)
        
        self.drift_history.append(drift_report)
        self._save_drift_report(drift_report)
        self._generate_drift_plots(new_data)
        
        return drift_report
    
    def _save_drift_report(self, report):
        """Save drift report to CSV"""
        os.makedirs('logs', exist_ok=True)
        df = pd.DataFrame([{
            'timestamp': report['timestamp'],
            'overall_drift_score': report['overall_drift_score']
        }])
        
        if os.path.exists('logs/drift_report.csv'):
            df.to_csv('logs/drift_report.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('logs/drift_report.csv', index=False)
    
    def _generate_drift_plots(self, new_data):
        """Generate comparison plots for drifting features"""
        os.makedirs('outputs/drift_plots', exist_ok=True)
        
        for feature in self.features:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.training_data[feature], label='Training', alpha=0.5)
            sns.histplot(new_data[feature], label='New Data', alpha=0.5)
            plt.title(f'Distribution Comparison: {feature}')
            plt.legend()
            plt.savefig(f'outputs/drift_plots/{feature}_comparison.png')
            plt.close()
    
    def get_drift_summary(self):
        """Get summary of drift detection history"""
        if not self.drift_history:
            return "No drift detection history available"
            
        latest = self.drift_history[-1]
        drifting_features = [
            feat for feat, report in latest['features'].items()
            if report['drift_detected']
        ]
        
        return {
            'timestamp': latest['timestamp'],
            'overall_drift_score': latest['overall_drift_score'],
            'drifting_features': drifting_features,
            'total_features_checked': len(self.features)
        }

def monitor_api_data(api_data_path='api_simulation_data.csv'):
    """Monitor drift in API simulation data"""
    detector = DriftDetector()
    new_data = pd.read_csv(api_data_path)
    
    # Drop target column if present
    if 'Class' in new_data.columns:
        new_data = new_data.drop('Class', axis=1)
    
    drift_report = detector.detect_drift(new_data)
    return detector.get_drift_summary()

if __name__ == "__main__":
    # Example usage
    summary = monitor_api_data()
    print("Drift Detection Summary:")
    print(f"Overall Drift Score: {summary['overall_drift_score']:.4f}")
    print(f"Drifting Features: {summary['drifting_features']}")
    print(f"Total Features Checked: {summary['total_features_checked']}") 