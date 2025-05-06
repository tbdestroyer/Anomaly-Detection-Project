import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from anomaly_detection.monitoring import (
    DataMonitor, ModelMonitor, ResourceMonitor,
    RetrainingTrigger, MLOpsMetrics
)
from datetime import datetime, timedelta
import json
import os

# Set page config
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Set style
plt.style.use('seaborn')
sns.set_palette('husl')

# Create necessary directories if they don't exist
os.makedirs('outputs/analysis', exist_ok=True)
os.makedirs('outputs/metrics', exist_ok=True)
os.makedirs('outputs/monitoring', exist_ok=True)
os.makedirs('models/versions', exist_ok=True)

# Initialize components
data_monitor = DataMonitor("creditcard.csv")
model_monitor = ModelMonitor()
resource_monitor = ResourceMonitor()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Quality", "Model Performance", "System Metrics", "Versioning"])

# Data Quality Page
if page == "Data Quality":
    st.title("Data Quality Dashboard")
    
    # Time range selector
    st.sidebar.header("Time Range")
    time_range = st.sidebar.selectbox(
        "Select Time Range",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
    )
    
    # Load data quality metrics
    try:
        metrics_df = pd.read_csv('outputs/analysis/analysis_metrics_log.csv')
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        
        # Filter by time range
        if time_range == "Last 24 hours":
            metrics_df = metrics_df[metrics_df['timestamp'] > datetime.now() - timedelta(days=1)]
        elif time_range == "Last 7 days":
            metrics_df = metrics_df[metrics_df['timestamp'] > datetime.now() - timedelta(days=7)]
        elif time_range == "Last 30 days":
            metrics_df = metrics_df[metrics_df['timestamp'] > datetime.now() - timedelta(days=30)]
        
        if len(metrics_df) == 0:
            st.warning("No data available for the selected time range.")
        else:
            # Data Distribution
            st.header("Data Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Types")
                fig, ax = plt.subplots()
                metrics_df.iloc[-1][['num_numeric', 'num_categorical']].plot(kind='bar', ax=ax)
                plt.title('Number of Features by Type')
                plt.xlabel('Feature Type')
                plt.ylabel('Count')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Missing Values")
                fig, ax = plt.subplots()
                metrics_df['total_missing'].plot(ax=ax)
                plt.title('Missing Values Over Time')
                plt.xlabel('Time')
                plt.ylabel('Missing Count')
                st.pyplot(fig)
            
            # Outlier Analysis
            st.header("Outlier Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Outlier Features")
                fig, ax = plt.subplots()
                metrics_df['num_outlier_features'].plot(ax=ax)
                plt.title('Number of Features with Outliers')
                plt.xlabel('Time')
                plt.ylabel('Count')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Average Outlier Percentage")
                fig, ax = plt.subplots()
                metrics_df['avg_outlier_percentage'].plot(ax=ax)
                plt.title('Average Outlier Percentage Over Time')
                plt.xlabel('Time')
                plt.ylabel('Percentage')
                st.pyplot(fig)
    
    except FileNotFoundError:
        st.error("Data quality metrics not found. Please ensure the analysis pipeline has been run.")
    except Exception as e:
        st.error(f"Error loading data quality metrics: {str(e)}")

# Model Performance Page
elif page == "Model Performance":
    st.title("Model Performance Dashboard")
    
    # Model selector
    models = ['isolation_forest', 'oneclass_svm', 'elliptic_envelope', 'lof', 'autoencoder', 'ensemble_light', 'ensemble_full']
    selected_model = st.sidebar.selectbox("Select Model", models)
    
    # Metric selector
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    selected_metric = st.sidebar.selectbox("Select Metric", metrics)
    
    try:
        # Load model metrics
        metrics_df = pd.read_csv('outputs/metrics/metrics_log.csv')
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        
        # Filter for selected model
        model_metrics = metrics_df[metrics_df['model'] == selected_model]
        
        if len(model_metrics) == 0:
            st.warning(f"No metrics available for {selected_model}")
        else:
            # Display metrics
            st.header("Model Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            latest_metrics = model_metrics.iloc[-1]
            col1.metric("Accuracy", f"{latest_metrics['accuracy']:.3f}")
            col2.metric("Precision", f"{latest_metrics['precision']:.3f}")
            col3.metric("Recall", f"{latest_metrics['recall']:.3f}")
            col4.metric("F1 Score", f"{latest_metrics['f1']:.3f}")
            
            # Metric trend
            st.subheader(f"{selected_metric.capitalize()} Trend")
            fig, ax = plt.subplots()
            model_metrics[selected_metric].plot(ax=ax)
            plt.title(f'{selected_metric.capitalize()} Over Time')
            plt.xlabel('Time')
            plt.ylabel(selected_metric.capitalize())
            st.pyplot(fig)
            
            # Confusion Matrix
            st.header("Confusion Matrix")
            try:
                with open(f'outputs/metrics/{selected_model}_predictions.json', 'r') as f:
                    predictions = json.load(f)
                    cm = confusion_matrix(predictions['y_true'], predictions['y_pred'])
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.title(f'Confusion Matrix - {selected_model}')
                    st.pyplot(fig)
            except FileNotFoundError:
                st.warning("Confusion matrix not available for this model")
            except Exception as e:
                st.error(f"Error loading confusion matrix: {str(e)}")
    
    except FileNotFoundError:
        st.error("Model metrics not found. Please ensure the evaluation pipeline has been run.")
    except Exception as e:
        st.error(f"Error loading model metrics: {str(e)}")

# System Metrics Page
elif page == "System Metrics":
    st.title("System Metrics Dashboard")
    
    # Time range selector
    st.sidebar.header("Time Range")
    time_range = st.sidebar.selectbox(
        "Select Time Range",
        ["Last hour", "Last 6 hours", "Last 24 hours", "All time"]
    )
    
    try:
        # Load system metrics
        metrics_df = pd.read_csv('outputs/monitoring/system_metrics.csv')
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        
        # Filter by time range
        if time_range == "Last hour":
            metrics_df = metrics_df[metrics_df['timestamp'] > datetime.now() - timedelta(hours=1)]
        elif time_range == "Last 6 hours":
            metrics_df = metrics_df[metrics_df['timestamp'] > datetime.now() - timedelta(hours=6)]
        elif time_range == "Last 24 hours":
            metrics_df = metrics_df[metrics_df['timestamp'] > datetime.now() - timedelta(hours=24)]
        
        if len(metrics_df) == 0:
            st.warning("No data available for the selected time range.")
        else:
            # Resource Usage
            st.header("Resource Usage")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("CPU Usage")
                fig, ax = plt.subplots()
                metrics_df['cpu_usage'].plot(ax=ax)
                plt.title('CPU Usage Over Time')
                plt.xlabel('Time')
                plt.ylabel('Usage (%)')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Memory Usage")
                fig, ax = plt.subplots()
                metrics_df['memory_usage'].plot(ax=ax)
                plt.title('Memory Usage Over Time')
                plt.xlabel('Time')
                plt.ylabel('Usage (%)')
                st.pyplot(fig)
            
            # Latency
            st.header("Latency Metrics")
            try:
                with open('outputs/monitoring/latency_metrics.json', 'r') as f:
                    latency_data = json.load(f)
                    fig, ax = plt.subplots()
                    sns.histplot(latency_data['latencies'], kde=True, ax=ax)
                    plt.title('Latency Distribution')
                    plt.xlabel('Latency (ms)')
                    plt.ylabel('Count')
                    st.pyplot(fig)
            except FileNotFoundError:
                st.warning("Latency data not available")
            except Exception as e:
                st.error(f"Error loading latency data: {str(e)}")
    
    except FileNotFoundError:
        st.error("System metrics not found. Please ensure the monitoring pipeline has been run.")
    except Exception as e:
        st.error(f"Error loading system metrics: {str(e)}")

# Versioning Page
elif page == "Versioning":
    st.title("Versioning Dashboard")
    
    try:
        # Version selector
        versions = sorted([f.stem for f in Path('models/versions').glob('*.json')])
        
        if not versions:
            st.warning("No model versions found. Please ensure models have been trained and versioned.")
        else:
            selected_version = st.sidebar.selectbox("Select Version", versions)
            
            # Load version data
            with open(f'models/versions/{selected_version}.json', 'r') as f:
                version_data = json.load(f)
            
            # Version Timeline
            st.header("Version Timeline")
            fig, ax = plt.subplots()
            timestamps = [datetime.fromisoformat(v['timestamp']) for v in version_data]
            version_nums = [v['version'] for v in version_data]
            ax.scatter(timestamps, version_nums)
            plt.title('Model Version Timeline')
            plt.xlabel('Time')
            plt.ylabel('Version')
            st.pyplot(fig)
            
            # Version Details
            st.header("Version Details")
            st.json(version_data)
            
            # Performance Comparison
            st.header("Performance Comparison")
            if len(versions) >= 2:
                latest = version_data[-1]
                previous = version_data[-2]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Latest Version")
                    st.write(f"Version: {latest['version']}")
                    st.write(f"Timestamp: {latest['timestamp']}")
                    st.write(f"Accuracy: {latest.get('accuracy', 'N/A')}")
                    st.write(f"F1 Score: {latest.get('f1', 'N/A')}")
                
                with col2:
                    st.subheader("Previous Version")
                    st.write(f"Version: {previous['version']}")
                    st.write(f"Timestamp: {previous['timestamp']}")
                    st.write(f"Accuracy: {previous.get('accuracy', 'N/A')}")
                    st.write(f"F1 Score: {previous.get('f1', 'N/A')}")
    
    except FileNotFoundError:
        st.error("Version data not found. Please ensure model versioning has been set up.")
    except Exception as e:
        st.error(f"Error loading version data: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Anomaly Detection Dashboard - Monitoring and Evaluation") 