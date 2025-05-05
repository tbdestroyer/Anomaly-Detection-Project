import pandas as pd
import numpy as np
from load_models import load_all_models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings("ignore")  # To suppress TensorFlow and sklearn warnings for clean output

def predict_autoencoder(ae_model, ae_threshold, X):
    reconstructed = ae_model.predict(X)
    mse = np.mean(np.square(X - reconstructed), axis=1)
    return (mse > ae_threshold).astype(int)

def tune_ensemble_medium(input_path='api_simulation_data.csv'):
    print("🔹 Loading models...")
    models = load_all_models()

    print(f"🔹 Reading input data from {input_path}...")
    data = pd.read_csv(input_path)

    y_true = data['Class']
    X = data.drop('Class', axis=1)

    print("🔹 Generating individual model predictions...")
    svm_pred = np.where(models['svm'].predict(X) == 1, 0, 1)
    elliptic_pred = np.where(models['elliptic'].predict(X) == 1, 0, 1)
    ae_pred = predict_autoencoder(models['autoencoder'], models['ae_threshold'], X)

    predictions = np.vstack([svm_pred, elliptic_pred, ae_pred]).T  # Shape: (n_samples, 3)

    best_auc = 0
    best_threshold = None
    best_report = ""

    print("\n🔹 Tuning voting thresholds...")
    for threshold in [1, 2, 3]:
        combined_pred = (predictions.sum(axis=1) >= threshold).astype(int)
        auc = roc_auc_score(y_true, combined_pred)
        report = classification_report(y_true, combined_pred, digits=4)

        print(f"\nThreshold: {threshold}")
        print(f"ROC AUC: {auc:.4f}")
        print(report)

        if auc > best_auc:
            best_auc = auc
            best_threshold = threshold
            best_report = report

    print("\n✅ Best Threshold Found!")
    print(f"Threshold: {best_threshold} | ROC AUC: {best_auc:.4f}")
    print(best_report)

    # Optionally, save best config
    with open('outputs/ensemble_medium_tuning_results.txt', 'w') as f:
        f.write(f"Best Threshold: {best_threshold}\n")
        f.write(f"ROC AUC: {best_auc:.4f}\n\n")
        f.write(best_report)

if __name__ == "__main__":
    tune_ensemble_medium()
