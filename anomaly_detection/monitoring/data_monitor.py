import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

class DataMonitor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.output_dir = 'outputs/monitoring'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def validate_data(self):
        """Perform basic data validation checks"""
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'numeric_stats': self._calculate_numeric_stats(),
            'categorical_stats': self._calculate_categorical_stats()
        }
        
        # Save validation results
        with open(f'{self.output_dir}/validation_results.json', 'w') as f:
            json.dump(validation_results, f)
            
        return validation_results
        
    def detect_drift(self, reference_data=None):
        """Detect data drift using statistical tests"""
        if reference_data is None:
            reference_data = self.data
            
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'ks_tests': {},
            'chi2_tests': {}
        }
        
        # Compare numeric columns using KS test
        for col in self.numeric_cols:
            ks_stat, p_value = stats.ks_2samp(
                reference_data[col].dropna(),
                self.data[col].dropna()
            )
            drift_results['ks_tests'][col] = {
                'statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < 0.05
            }
            
        # Compare categorical columns using chi-square test
        for col in self.categorical_cols:
            if len(self.data[col].unique()) > 1:
                contingency = pd.crosstab(
                    reference_data[col],
                    self.data[col]
                )
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                drift_results['chi2_tests'][col] = {
                    'statistic': chi2,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05
                }
                
        # Save drift results
        with open(f'{self.output_dir}/drift_results.json', 'w') as f:
            json.dump(drift_results, f)
            
        return drift_results
        
    def visualize_validation_results(self):
        """Create visualizations for validation results"""
        # Load validation results
        with open(f'{self.output_dir}/validation_results.json', 'r') as f:
            results = json.load(f)
            
        # Create missing values plot
        plt.figure(figsize=(12, 6))
        missing_values = pd.Series(results['missing_values'])
        missing_values.plot(kind='bar')
        plt.title('Missing Values by Column')
        plt.xlabel('Column')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/missing_values.png')
        plt.close()
        
        # Create numeric stats plot
        numeric_stats = pd.DataFrame(results['numeric_stats'])
        plt.figure(figsize=(12, 6))
        numeric_stats.T.plot(kind='box')
        plt.title('Numeric Column Statistics')
        plt.xlabel('Statistic')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/numeric_stats.png')
        plt.close()
        
    def visualize_drift_results(self):
        """Create visualizations for drift detection results"""
        # Load drift results
        with open(f'{self.output_dir}/drift_results.json', 'r') as f:
            results = json.load(f)
            
        # Create KS test results plot
        ks_results = pd.DataFrame(results['ks_tests']).T
        plt.figure(figsize=(12, 6))
        ks_results['p_value'].plot(kind='bar')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.title('Kolmogorov-Smirnov Test Results')
        plt.xlabel('Column')
        plt.ylabel('p-value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ks_test_results.png')
        plt.close()
        
        # Create chi-square test results plot
        if results['chi2_tests']:
            chi2_results = pd.DataFrame(results['chi2_tests']).T
            plt.figure(figsize=(12, 6))
            chi2_results['p_value'].plot(kind='bar')
            plt.axhline(y=0.05, color='r', linestyle='--')
            plt.title('Chi-Square Test Results')
            plt.xlabel('Column')
            plt.ylabel('p-value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/chi2_test_results.png')
            plt.close()
            
    def _calculate_numeric_stats(self):
        """Calculate statistics for numeric columns"""
        stats_dict = {}
        for col in self.numeric_cols:
            stats_dict[col] = {
                'mean': self.data[col].mean(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'skew': self.data[col].skew(),
                'kurtosis': self.data[col].kurtosis()
            }
        return stats_dict
        
    def _calculate_categorical_stats(self):
        """Calculate statistics for categorical columns"""
        stats_dict = {}
        for col in self.categorical_cols:
            stats_dict[col] = {
                'unique_values': len(self.data[col].unique()),
                'most_common': self.data[col].mode().iloc[0],
                'missing_count': self.data[col].isnull().sum()
            }
        return stats_dict 