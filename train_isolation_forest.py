import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_isolation_forest(train_path, test_path, contamination=0.01, random_state=42, output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading datasets...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']
    
    print("\nTraining Isolation Forest model with benchmarking...")
    process = psutil.Process()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)

    start_time = time.perf_counter()
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    end_time = time.perf_counter()

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)

    training_time = end_time - start_time
    cpu_usage = cpu_after - cpu_before
    mem_usage = mem_after - mem_before

    # Predictions
    y_pred_train = np.where(iso_forest.predict(X_train) == 1, 0, 1)
    y_pred_test = np.where(iso_forest.predict(X_test) == 1, 0, 1)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    # Confusion Matrices Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True, fmt='d', cmap='Blues')
    plt.title('Training Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Blues')
    plt.title('Testing Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # Save model
    model_path = os.path.join(output_dir, 'isolation_forest_model.joblib')
    joblib.dump(iso_forest, model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # in MB

    print(f"\nModel saved as '{model_path}'")

    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Isolation Forest Training Benchmark Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Training Time: {training_time:.2f} sec", ln=True)
    pdf.cell(0, 10, f"CPU Usage Increase: {cpu_usage:.2f} %", ln=True)
    pdf.cell(0, 10, f"Memory Usage Increase: {mem_usage:.2f} MB", ln=True)
    pdf.cell(0, 10, f"Model Size: {model_size:.2f} MB", ln=True)
    pdf.ln(10)

    pdf.cell(0, 10, "Test Set Performance (F1-Score):", ln=True)
    pdf.cell(0, 10, f" - F1-Score (Class 1): {test_report['1']['f1-score']:.4f}", ln=True)
    pdf.cell(0, 10, f" - F1-Score (Class 0): {test_report['0']['f1-score']:.4f}", ln=True)

    pdf.add_page()
    pdf.image(cm_path, x=10, y=20, w=180)

    pdf_path = os.path.join(output_dir, 'training_report.pdf')
    pdf.output(pdf_path)

    print(f"\nTraining report saved as '{pdf_path}'")

if __name__ == "__main__":
    train_isolation_forest('creditcard_train.csv', 'creditcard_test.csv')
