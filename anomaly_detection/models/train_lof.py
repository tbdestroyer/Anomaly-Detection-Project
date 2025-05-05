import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_lof(train_path='creditcard_train.csv', output_dir='outputs'):
    print("🔹 Loading pre-scaled training data...")
    data = pd.read_csv(train_path)
    X_train = data.drop('Class', axis=1)
    y_train = data['Class']

    print("🔹 Training Local Outlier Factor (LOF)...")
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
    model.fit(X_train)

    print("✅ Training completed! Generating predictions...")
    y_pred = np.where(model.predict(X_train) == 1, 0, 1)  # 1 = normal, -1 = anomaly

    print("🔹 Calculating evaluation metrics...")
    report = classification_report(y_train, y_pred, digits=4)
    cm = confusion_matrix(y_train, y_pred)
    fpr, tpr, _ = roc_curve(y_train, y_pred)
    roc_auc = auc(fpr, tpr)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save classification report
    with open(os.path.join(output_dir, 'lof_classification_report.txt'), 'w') as f:
        f.write(report)
        f.write(f"\nROC AUC: {roc_auc:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('LOF - Confusion Matrix (Train Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'lof_confusion_matrix.png'))
    plt.close()

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LOF - ROC Curve (Train Set)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'lof_roc_curve.png'))
    plt.close()

    # Save model
    joblib.dump(model, os.path.join(output_dir, 'lof_model.joblib'))

    print(f"✅ LOF model and metrics saved in '{output_dir}'")

if __name__ == "__main__":
    train_lof()
