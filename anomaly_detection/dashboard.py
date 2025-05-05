
import streamlit as st
import pandas as pd
import ast
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("🚀 Anomaly Detection - Unified Dashboard")

option = st.sidebar.radio(
    "Select View:",
    ["📄 Classification Metrics", "📊 Confusion Matrix", "📈 ROC Curve", "⏱️ Latency Distribution", "⚡ Load Test Metrics"]
)

def load_classification_data():
    data = pd.read_csv("prediction_results_log.csv")
    data['y_true'] = data['y_true'].apply(ast.literal_eval)
    data['y_pred'] = data['y_pred'].apply(ast.literal_eval)
    return data

def load_requests_data():
    return pd.read_csv("logs/requests.csv")

if option == "📄 Classification Metrics":
    try:
        data = load_classification_data()
        y_true_flat = [item for sublist in data['y_true'] for item in sublist]
        y_pred_flat = [item for sublist in data['y_pred'] for item in sublist]
        report = classification_report(y_true_flat, y_pred_flat, output_dict=True)
        roc_auc_value = auc(*roc_curve(y_true_flat, y_pred_flat)[:2])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precision (Fraud)", round(report['1']['precision'], 4))
        col2.metric("Recall (Fraud)", round(report['1']['recall'], 4))
        col3.metric("F1-Score (Fraud)", round(report['1']['f1-score'], 4))
        col4.metric("ROC AUC", round(roc_auc_value, 4))
    except:
        st.error("Classification log not found or invalid.")

elif option == "📊 Confusion Matrix":
    try:
        data = load_classification_data()
        y_true_flat = [item for sublist in data['y_true'] for item in sublist]
        y_pred_flat = [item for sublist in data['y_pred'] for item in sublist]
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        fig, ax = plt.subplots(figsize=(2.5,2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        centered = st.columns([1, 2, 1])
        with centered[1]:
            st.pyplot(fig, use_container_width=False)
    except:
        st.error("Classification log not found or invalid.")

elif option == "📈 ROC Curve":
    try:
        data = load_classification_data()
        y_true_flat = [item for sublist in data['y_true'] for item in sublist]
        y_pred_flat = [item for sublist in data['y_pred'] for item in sublist]
        fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
        fig, ax = plt.subplots(figsize=(2.5,2))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle='--')
        centered = st.columns([1, 2, 1])
        with centered[1]:
            st.pyplot(fig, use_container_width=False)
    except:
        st.error("Classification log not found or invalid.")

elif option == "⏱️ Latency Distribution":
    try:
        data = load_classification_data()
        fig, ax = plt.subplots(figsize=(3,2))
        data['latency_ms'].plot.hist(bins=30, ax=ax, color='skyblue')
        centered = st.columns([1, 2, 1])
        with centered[1]:
            st.pyplot(fig, use_container_width=False)
    except:
        st.error("Classification log not found or invalid.")

elif option == "⚡ Load Test Metrics":
    try:
        df = load_requests_data()
        row = df.iloc[0]
        total_rows = row['Request Count'] * 3
        failure_rate = round((row['Failure Count'] / row['Request Count']) * 100, 2)
        st.metric("Total Rows Processed", int(total_rows))
        st.metric("Failure Rate", f"{failure_rate}%")
        percentiles = {'50%': row['50%'], '75%': row['75%'], '95%': row['95%'], '99%': row['99%']}
        fig, ax = plt.subplots(figsize=(3,2))
        ax.bar(percentiles.keys(), percentiles.values(), color='teal')
        centered = st.columns([1, 2, 1])
        with centered[1]:
            st.pyplot(fig, use_container_width=False)
    except:
        st.error("Load test data not found. Run Locust first.")
