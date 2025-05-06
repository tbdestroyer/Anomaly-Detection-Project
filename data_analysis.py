import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, data_path):
        """
        Initialize the data analyzer with a dataset.
        
        Args:
            data_path (str): Path to the data file
        """
        self.data = pd.read_csv(data_path)
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Create output directory
        self.output_dir = Path('outputs/analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analysis results
        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'summary_stats': {},
            'correlations': {},
            'distributions': {},
            'missing_values': {},
            'outliers': {}
        }
    
    def analyze_data(self):
        """Perform comprehensive data analysis."""
        self._calculate_summary_statistics()
        self._calculate_correlations()
        self._analyze_distributions()
        self._check_missing_values()
        self._detect_outliers()
        
        # Save analysis results
        self._save_analysis_results()
        
        # Log analysis metrics to CSV
        self._log_analysis_metrics_to_csv()
        
        return self.analysis_results
    
    def _calculate_summary_statistics(self):
        """Calculate summary statistics for numeric columns."""
        for col in self.numeric_cols:
            self.analysis_results['summary_stats'][col] = {
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'skew': self.data[col].skew(),
                'kurtosis': self.data[col].kurtosis()
            }
    
    def _calculate_correlations(self):
        """Calculate correlations between numeric columns."""
        corr_matrix = self.data[self.numeric_cols].corr()
        self.analysis_results['correlations'] = corr_matrix.to_dict()
        
        # Save correlation matrix plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        output_file = self.output_dir / 'correlations' / 'correlation_matrix.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()
    
    def _analyze_distributions(self):
        """Analyze distributions of numeric columns."""
        for col in self.numeric_cols:
            # Calculate distribution statistics
            self.analysis_results['distributions'][col] = {
                'shapiro_test': stats.shapiro(self.data[col])[1],
                'anderson_test': stats.anderson(self.data[col])[0],
                'ks_test': stats.kstest(self.data[col], 'norm')[1]
            }
            
            # Save distribution plot
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            
            output_file = self.output_dir / 'distributions' / f'{col}_distribution.png'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file)
            plt.close()
    
    def _check_missing_values(self):
        """Check for missing values in the dataset."""
        missing_values = self.data.isnull().sum()
        self.analysis_results['missing_values'] = missing_values[missing_values > 0].to_dict()
        
        if len(self.analysis_results['missing_values']) > 0:
            # Save missing values plot
            plt.figure(figsize=(10, 6))
            pd.Series(self.analysis_results['missing_values']).plot(kind='bar')
            plt.title('Missing Values by Feature')
            plt.xlabel('Feature')
            plt.ylabel('Missing Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            output_file = self.output_dir / 'missing_values.png'
            plt.savefig(output_file)
            plt.close()
    
    def _detect_outliers(self):
        """Detect outliers in numeric columns using IQR method."""
        for col in self.numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[
                (self.data[col] < lower_bound) | 
                (self.data[col] > upper_bound)
            ][col]
            
            self.analysis_results['outliers'][col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.data)) * 100,
                'values': outliers.tolist()
            }
    
    def _save_analysis_results(self):
        """Save analysis results to a JSON file."""
        output_file = self.output_dir / 'data_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=4)
    
    def _log_analysis_metrics_to_csv(self):
        """Log analysis metrics to CSV file for tracking over time."""
        csv_file = self.output_dir / 'analysis_metrics_log.csv'
        
        # Create metrics row
        metrics_row = {
            'timestamp': self.analysis_results['timestamp'],
            'num_features': len(self.numeric_cols) + len(self.categorical_cols),
            'num_numeric': len(self.numeric_cols),
            'num_categorical': len(self.categorical_cols),
            'total_missing': sum(self.analysis_results['missing_values'].values()),
            'num_outlier_features': len(self.analysis_results['outliers']),
            'avg_outlier_percentage': np.mean([
                stats['percentage'] 
                for stats in self.analysis_results['outliers'].values()
            ])
        }
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics_row])
        
        # Append to existing CSV or create new one
        if csv_file.exists():
            metrics_df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(csv_file, index=False)
    
    def get_analysis_history(self):
        """Retrieve historical analysis results."""
        output_file = self.output_dir / 'data_analysis_results.json'
        if output_file.exists():
            with open(output_file, 'r') as f:
                return json.load(f)
        return None
    
    def visualize_analysis(self):
        """Generate comprehensive visualization of analysis results."""
        # Create a summary figure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Missing Values
        plt.subplot(2, 2, 1)
        pd.Series(self.analysis_results['missing_values']).plot(kind='bar')
        plt.title('Missing Values by Feature')
        plt.xlabel('Feature')
        plt.ylabel('Missing Count')
        plt.xticks(rotation=45)
        
        # Plot 2: Outlier Distribution
        plt.subplot(2, 2, 2)
        outlier_percentages = {
            col: stats['percentage'] 
            for col, stats in self.analysis_results['outliers'].items()
        }
        pd.Series(outlier_percentages).plot(kind='bar')
        plt.title('Outlier Percentage by Feature')
        plt.xlabel('Feature')
        plt.ylabel('Outlier Percentage')
        plt.xticks(rotation=45)
        
        # Plot 3: Feature Distributions (first 4 features)
        plt.subplot(2, 2, 3)
        for col in list(self.numeric_cols)[:4]:
            sns.kdeplot(self.data[col], label=col)
        plt.title('Feature Distributions')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        
        # Plot 4: Correlation Heatmap
        plt.subplot(2, 2, 4)
        corr_matrix = pd.DataFrame(self.analysis_results['correlations'])
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        # Save the summary plot
        output_file = self.output_dir / 'analysis_summary.png'
        plt.savefig(output_file)
        plt.close()
