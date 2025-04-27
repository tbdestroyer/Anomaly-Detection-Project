import pandas as pd
import numpy as np
from load_models import load_all_models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os


def predict_autoencoder(ae_model, ae_threshold, X):
    reconstructed = ae_model.predict(X)
    mse = np.mean(np.square(X - reconstructed), axis=1)
    return (mse > ae_threshold).astype(int)
def ensemble_light(input_path='api_simulation_data.csv', output_dir='outputs'):
    print("🔹 Loading models...")
    models = load_all_models()

    print(f"🔹 Reading input data from {input_path}...")
    data = pd.read_csv(input_path)

    if 'Class' in data.columns:
        y_true = data['Class']
        data = data.drop('Class', axis=1)
    else:
        y_true = None

    # Predictions
    iso_pred = np.where(models['isolation_forest'].predict(data) == 1, 0, 1)
    svm_pred = np.where(models['svm'].predict(data) == 1, 0, 1)
    ae_pred = predict_autoencoder(models['autoencoder'], models['ae_threshold'], data)

    combined_pred = ((iso_pred + svm_pred + ae_pred) >= 2).astype(int)

    print(f"✅ Ensemble predictions completed. Sample output:\n{combined_pred[:10]}")

    if y_true is not None:
        print("🔹 Calculating evaluation metrics...")
        report = classification_report(y_true, combined_pred, digits=4)
        cm = confusion_matrix(y_true, combined_pred)
        fpr, tpr, _ = roc_curve(y_true, combined_pred)
        roc_auc = auc(fpr, tpr)

        # Print report
        print(report)
        print(f"ROC AUC: {roc_auc:.4f}")

        # Save metrics
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'ensemble_light_report.txt'), 'w') as f:
            f.write(report)
            f.write(f"\nROC AUC: {roc_auc:.4f}")

        # Plot confusion matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Light - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, 'ensemble_light_confusion_matrix.png'))
        plt.close()

        # Plot ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble Light - ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'ensemble_light_roc_curve.png'))
        plt.close()

        print(f"✅ Metrics saved in '{output_dir}'")

    return combined_pred

if __name__ == "__main__":
    ensemble_light()
