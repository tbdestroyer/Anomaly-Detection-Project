import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from anomaly_detection.visualization.metrics_visualizer import MetricsVisualizer

class EffectivenessDashboard:
    def __init__(self):
        self.visualizer = MetricsVisualizer()
        self.setup_page()
        
    def setup_page(self):
        st.set_page_config(layout="wide", page_title="MLOps Effectiveness Dashboard")
        st.title("MLOps System Effectiveness Dashboard")
        
    def run(self):
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Select View",
            ["Data Quality", "Model Performance", "Infrastructure", "Monitoring"]
        )
        
        if page == "Data Quality":
            self.show_data_quality()
        elif page == "Model Performance":
            self.show_model_performance()
        elif page == "Infrastructure":
            self.show_infrastructure()
        else:
            self.show_monitoring()
            
    def show_data_quality(self):
        st.header("Data Quality Metrics")
        
        # Data Drift Section
        st.subheader("Data Drift Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(self.visualizer.visualize_data_drift(
                self._load_sample_data('before'),
                self._load_sample_data('after'),
                ['feature1', 'feature2', 'feature3', 'feature4']
            ))
            
        with col2:
            st.plotly_chart(self.visualizer.visualize_data_validation(
                self._load_validation_data('before'),
                self._load_validation_data('after')
            ))
            
        # SMOTE/Sampling Impact
        st.subheader("SMOTE/Sampling Impact")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(self.visualizer.visualize_sampling_impact(
                self._load_sample_data('before'),
                self._load_sample_data('after'),
                'target'
            ))
            
        with col2:
            self._show_sampling_metrics()
            
    def show_model_performance(self):
        st.header("Model Performance Metrics")
        
        # Model Monitoring
        st.subheader("Model Performance Over Time")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(self.visualizer.visualize_model_performance(
                self._load_performance_data()
            ))
            
        with col2:
            self._show_performance_alerts()
            
        # Version Comparison
        st.subheader("Model Version Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(self.visualizer.visualize_version_comparison(
                self._load_version_metrics()
            ))
            
        with col2:
            self._show_ab_test_results()
            
    def show_infrastructure(self):
        st.header("Infrastructure Metrics")
        
        # CI/CD Pipeline
        st.subheader("CI/CD Pipeline Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            self._show_pipeline_metrics()
            
        with col2:
            self._show_deployment_metrics()
            
        # Resource Usage
        st.subheader("Resource Usage")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(self.visualizer.visualize_resource_usage(
                self._load_resource_data()
            ))
            
        with col2:
            self._show_health_checks()
            
    def show_monitoring(self):
        st.header("Monitoring Metrics")
        
        # Logging and Alerts
        st.subheader("Logging and Alert Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(self.visualizer.visualize_logging_metrics(
                self._load_log_data()
            ))
            
        with col2:
            self._show_alert_summary()
            
    def _show_sampling_metrics(self):
        st.metric("Class Balance Before", "60/40")
        st.metric("Class Balance After", "50/50")
        st.metric("F1 Score Improvement", "+15%")
        st.metric("AUC Improvement", "+8%")
        
    def _show_performance_alerts(self):
        st.subheader("Performance Alerts")
        alerts = [
            {"timestamp": "2024-03-20 10:00", "metric": "Accuracy", "change": "-5%", "status": "warning"},
            {"timestamp": "2024-03-20 09:30", "metric": "F1 Score", "change": "-3%", "status": "warning"},
            {"timestamp": "2024-03-20 09:00", "metric": "Latency", "change": "+20ms", "status": "critical"}
        ]
        
        for alert in alerts:
            st.warning(f"{alert['timestamp']} - {alert['metric']}: {alert['change']}")
            
    def _show_ab_test_results(self):
        st.subheader("A/B Test Results")
        results = {
            "Version A": {"accuracy": 0.85, "f1": 0.82},
            "Version B": {"accuracy": 0.88, "f1": 0.85}
        }
        
        for version, metrics in results.items():
            st.metric(f"{version} Accuracy", f"{metrics['accuracy']:.2%}")
            st.metric(f"{version} F1 Score", f"{metrics['f1']:.2%}")
            
    def _show_pipeline_metrics(self):
        st.subheader("Pipeline Metrics")
        metrics = {
            "Success Rate": "95%",
            "Average Duration": "15 min",
            "Failed Runs": "2",
            "Rollbacks": "1"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
            
    def _show_deployment_metrics(self):
        st.subheader("Deployment Metrics")
        metrics = {
            "Uptime": "99.9%",
            "Response Time": "200ms",
            "Error Rate": "0.1%",
            "Active Users": "1000"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
            
    def _show_health_checks(self):
        st.subheader("Health Checks")
        checks = [
            {"service": "API", "status": "healthy", "last_check": "2 min ago"},
            {"service": "Database", "status": "healthy", "last_check": "1 min ago"},
            {"service": "Cache", "status": "warning", "last_check": "5 min ago"}
        ]
        
        for check in checks:
            if check["status"] == "healthy":
                st.success(f"{check['service']} - {check['last_check']}")
            else:
                st.warning(f"{check['service']} - {check['last_check']}")
                
    def _show_alert_summary(self):
        st.subheader("Alert Summary")
        alerts = {
            "Critical": 2,
            "Warning": 5,
            "Info": 10
        }
        
        for level, count in alerts.items():
            st.metric(f"{level} Alerts", count)
            
    def _load_sample_data(self, period):
        # Sample data loading implementation
        return pd.DataFrame()
        
    def _load_validation_data(self, period):
        # Validation data loading implementation
        return pd.DataFrame()
        
    def _load_performance_data(self):
        # Performance data loading implementation
        return pd.DataFrame()
        
    def _load_version_metrics(self):
        # Version metrics loading implementation
        return []
        
    def _load_resource_data(self):
        # Resource data loading implementation
        return pd.DataFrame()
        
    def _load_log_data(self):
        # Log data loading implementation
        return pd.DataFrame()

if __name__ == "__main__":
    dashboard = EffectivenessDashboard()
    dashboard.run() 