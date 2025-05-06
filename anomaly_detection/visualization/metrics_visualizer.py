import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class MetricsVisualizer:
    def __init__(self):
        self.output_dir = 'outputs/visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def visualize_data_drift(self, before_data, after_data, feature_names):
        """Visualize data drift with before/after comparisons"""
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('Feature Distributions', 'Drift Metrics',
                                         'Population Stability Index', 'KL Divergence'))
        
        # Feature distributions
        for i, feature in enumerate(feature_names[:4]):  # Show first 4 features
            fig.add_trace(go.Histogram(x=before_data[feature], name=f'Before {feature}',
                                     opacity=0.75), row=1, col=1)
            fig.add_trace(go.Histogram(x=after_data[feature], name=f'After {feature}',
                                     opacity=0.75), row=1, col=1)
            
        # Drift metrics
        drift_metrics = self._calculate_drift_metrics(before_data, after_data)
        fig.add_trace(go.Scatter(x=list(range(len(drift_metrics))), 
                                y=drift_metrics['psi'],
                                name='PSI'), row=1, col=2)
        fig.add_trace(go.Scatter(x=list(range(len(drift_metrics))), 
                                y=drift_metrics['kl'],
                                name='KL Divergence'), row=1, col=2)
        
        # PSI heatmap
        psi_matrix = self._calculate_psi_matrix(before_data, after_data)
        fig.add_trace(go.Heatmap(z=psi_matrix, 
                                x=feature_names,
                                y=feature_names), row=2, col=1)
        
        # KL divergence heatmap
        kl_matrix = self._calculate_kl_matrix(before_data, after_data)
        fig.add_trace(go.Heatmap(z=kl_matrix,
                                x=feature_names,
                                y=feature_names), row=2, col=2)
        
        fig.update_layout(height=1000, width=1200, title_text="Data Drift Analysis")
        fig.write_html(f"{self.output_dir}/data_drift.html")
        
    def visualize_data_validation(self, before_checks, after_checks):
        """Visualize data validation results"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Validation Results', 'Missing Values',
                                         'Outliers', 'Data Quality Score'))
        
        # Validation results
        results = pd.DataFrame({
            'Before': before_checks,
            'After': after_checks
        })
        fig.add_trace(go.Bar(x=results.index, y=results['Before'], name='Before'),
                     row=1, col=1)
        fig.add_trace(go.Bar(x=results.index, y=results['After'], name='After'),
                     row=1, col=1)
        
        # Missing values heatmap
        fig.add_trace(go.Heatmap(z=before_checks['missing_values'],
                                x=before_checks.columns,
                                y=before_checks.index), row=1, col=2)
        
        # Outliers box plot
        fig.add_trace(go.Box(y=before_checks['outliers'], name='Before'),
                     row=2, col=1)
        fig.add_trace(go.Box(y=after_checks['outliers'], name='After'),
                     row=2, col=1)
        
        # Data quality score
        quality_scores = pd.DataFrame({
            'Before': before_checks['quality_score'],
            'After': after_checks['quality_score']
        })
        fig.add_trace(go.Scatter(x=quality_scores.index,
                                y=quality_scores['Before'],
                                name='Before'), row=2, col=2)
        fig.add_trace(go.Scatter(x=quality_scores.index,
                                y=quality_scores['After'],
                                name='After'), row=2, col=2)
        
        fig.update_layout(height=1000, width=1200, title_text="Data Validation Analysis")
        fig.write_html(f"{self.output_dir}/data_validation.html")
        
    def visualize_sampling_impact(self, before_data, after_data, target_col):
        """Visualize impact of SMOTE/sampling"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Class Distribution', 't-SNE Visualization',
                                         'PCA Visualization', 'Performance Impact'))
        
        # Class distribution
        before_counts = before_data[target_col].value_counts()
        after_counts = after_data[target_col].value_counts()
        fig.add_trace(go.Bar(x=before_counts.index, y=before_counts.values,
                            name='Before'), row=1, col=1)
        fig.add_trace(go.Bar(x=after_counts.index, y=after_counts.values,
                            name='After'), row=1, col=1)
        
        # t-SNE visualization
        tsne = TSNE(n_components=2)
        tsne_before = tsne.fit_transform(before_data.drop(target_col, axis=1))
        tsne_after = tsne.fit_transform(after_data.drop(target_col, axis=1))
        
        fig.add_trace(go.Scatter(x=tsne_before[:, 0], y=tsne_before[:, 1],
                                mode='markers', name='Before'), row=1, col=2)
        fig.add_trace(go.Scatter(x=tsne_after[:, 0], y=tsne_after[:, 1],
                                mode='markers', name='After'), row=1, col=2)
        
        # PCA visualization
        pca = PCA(n_components=2)
        pca_before = pca.fit_transform(before_data.drop(target_col, axis=1))
        pca_after = pca.fit_transform(after_data.drop(target_col, axis=1))
        
        fig.add_trace(go.Scatter(x=pca_before[:, 0], y=pca_before[:, 1],
                                mode='markers', name='Before'), row=2, col=1)
        fig.add_trace(go.Scatter(x=pca_after[:, 0], y=pca_after[:, 1],
                                mode='markers', name='After'), row=2, col=2)
        
        # Performance impact
        metrics = self._calculate_performance_metrics(before_data, after_data, target_col)
        fig.add_trace(go.Bar(x=list(metrics.keys()),
                            y=list(metrics.values()),
                            name='Performance Metrics'), row=2, col=2)
        
        fig.update_layout(height=1000, width=1200, title_text="Sampling Impact Analysis")
        fig.write_html(f"{self.output_dir}/sampling_impact.html")
        
    def visualize_model_performance(self, history_data):
        """Visualize model performance over time"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Accuracy Over Time', 'F1 Score Over Time',
                                         'AUC Over Time', 'Latency Distribution'))
        
        # Accuracy
        fig.add_trace(go.Scatter(x=history_data['timestamp'],
                                y=history_data['accuracy'],
                                name='Accuracy'), row=1, col=1)
        
        # F1 Score
        fig.add_trace(go.Scatter(x=history_data['timestamp'],
                                y=history_data['f1'],
                                name='F1 Score'), row=1, col=2)
        
        # AUC
        fig.add_trace(go.Scatter(x=history_data['timestamp'],
                                y=history_data['auc'],
                                name='AUC'), row=2, col=1)
        
        # Latency
        fig.add_trace(go.Box(y=history_data['latency'],
                            name='Latency'), row=2, col=2)
        
        fig.update_layout(height=1000, width=1200, title_text="Model Performance Over Time")
        fig.write_html(f"{self.output_dir}/model_performance.html")
        
    def visualize_version_comparison(self, version_metrics):
        """Visualize model version comparisons"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Version Metrics', 'A/B Test Results',
                                         'Confusion Matrix Comparison', 'ROC Curves'))
        
        # Version metrics
        metrics_df = pd.DataFrame(version_metrics)
        fig.add_trace(go.Bar(x=metrics_df['version'],
                            y=metrics_df['accuracy'],
                            name='Accuracy'), row=1, col=1)
        fig.add_trace(go.Bar(x=metrics_df['version'],
                            y=metrics_df['f1'],
                            name='F1 Score'), row=1, col=1)
        
        # A/B test results
        ab_results = self._get_ab_test_results()
        fig.add_trace(go.Bar(x=['Version A', 'Version B'],
                            y=[ab_results['version_a'], ab_results['version_b']],
                            name='A/B Test'), row=1, col=2)
        
        # Confusion matrices
        cm_before = confusion_matrix(version_metrics[0]['y_true'], version_metrics[0]['y_pred'])
        cm_after = confusion_matrix(version_metrics[-1]['y_true'], version_metrics[-1]['y_pred'])
        
        fig.add_trace(go.Heatmap(z=cm_before, name='Before'), row=2, col=1)
        fig.add_trace(go.Heatmap(z=cm_after, name='After'), row=2, col=2)
        
        # ROC curves
        for version in version_metrics:
            fpr, tpr, _ = roc_curve(version['y_true'], version['y_prob'])
            auc_score = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                    name=f'Version {version["version"]} (AUC={auc_score:.2f})'),
                         row=2, col=2)
        
        fig.update_layout(height=1000, width=1200, title_text="Model Version Comparison")
        fig.write_html(f"{self.output_dir}/version_comparison.html")
        
    def visualize_resource_usage(self, resource_data):
        """Visualize system resource usage"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('CPU Usage', 'Memory Usage',
                                         'Disk Usage', 'Network I/O'))
        
        # CPU Usage
        fig.add_trace(go.Scatter(x=resource_data['timestamp'],
                                y=resource_data['cpu_usage'],
                                name='CPU Usage'), row=1, col=1)
        
        # Memory Usage
        fig.add_trace(go.Scatter(x=resource_data['timestamp'],
                                y=resource_data['memory_usage'],
                                name='Memory Usage'), row=1, col=2)
        
        # Disk Usage
        fig.add_trace(go.Scatter(x=resource_data['timestamp'],
                                y=resource_data['disk_usage'],
                                name='Disk Usage'), row=2, col=1)
        
        # Network I/O
        fig.add_trace(go.Scatter(x=resource_data['timestamp'],
                                y=resource_data['network_io'],
                                name='Network I/O'), row=2, col=2)
        
        fig.update_layout(height=1000, width=1200, title_text="Resource Usage Over Time")
        fig.write_html(f"{self.output_dir}/resource_usage.html")
        
    def visualize_logging_metrics(self, log_data):
        """Visualize logging and alert metrics"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Log Volume', 'Alert Types',
                                         'Error Rate', 'Response Time'))
        
        # Log volume
        fig.add_trace(go.Histogram(x=log_data['timestamp'],
                                  y=log_data['volume'],
                                  name='Log Volume'), row=1, col=1)
        
        # Alert types
        alert_counts = log_data['alert_type'].value_counts()
        fig.add_trace(go.Bar(x=alert_counts.index,
                            y=alert_counts.values,
                            name='Alert Types'), row=1, col=2)
        
        # Error rate
        fig.add_trace(go.Scatter(x=log_data['timestamp'],
                                y=log_data['error_rate'],
                                name='Error Rate'), row=2, col=1)
        
        # Response time
        fig.add_trace(go.Box(y=log_data['response_time'],
                            name='Response Time'), row=2, col=2)
        
        fig.update_layout(height=1000, width=1200, title_text="Logging and Alert Metrics")
        fig.write_html(f"{self.output_dir}/logging_metrics.html")
        
    def _calculate_drift_metrics(self, before_data, after_data):
        """Calculate drift metrics between datasets"""
        metrics = {}
        for col in before_data.columns:
            metrics[col] = {
                'psi': self._calculate_psi(before_data[col], after_data[col]),
                'kl': self._calculate_kl(before_data[col], after_data[col])
            }
        return metrics
        
    def _calculate_psi(self, before, after):
        """Calculate Population Stability Index"""
        # Implementation of PSI calculation
        pass
        
    def _calculate_kl(self, before, after):
        """Calculate KL Divergence"""
        # Implementation of KL divergence calculation
        pass
        
    def _calculate_performance_metrics(self, before_data, after_data, target_col):
        """Calculate performance metrics for before/after comparison"""
        metrics = {
            'accuracy': accuracy_score(before_data[target_col], after_data[target_col]),
            'f1': f1_score(before_data[target_col], after_data[target_col]),
            'precision': precision_score(before_data[target_col], after_data[target_col]),
            'recall': recall_score(before_data[target_col], after_data[target_col])
        }
        return metrics
        
    def _get_ab_test_results(self):
        """Get A/B test results"""
        # Implementation to get A/B test results
        pass 