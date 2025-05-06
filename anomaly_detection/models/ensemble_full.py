import pandas as pd
import numpy as np
from load_models import load_all_models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import dump


def predict_autoencoder(ae_model, ae_threshold, X):
    reconstructed = ae_model.predict(X)
    mse = np.mean(np.square(X - reconstructed), axis=1)
    return (mse > ae_threshold).astype(int)


class MajorityVotingEnsemble(BaseEstimator):
    def __init__(self, models_dict):
        self.models = models_dict

    def predict(self, X):
        iso_pred = np.where(self.models['isolation_forest'].predict(X) == 1, 0, 1)
        svm_pred = np.where(self.models['svm'].predict(X) == 1, 0, 1)
        elliptic_pred = np.where(self.models['elliptic'].predict(X) == 1, 0, 1)
        lof_pred = np.where(self.models['lof'].predict(X) == 1, 0, 1)
        ae_pred = predict_autoencoder(self.models['autoencoder'], self.models['ae_threshold'], X)
        combined_pred = ((iso_pred + svm_pred + elliptic_pred + lof_pred + ae_pred) >= 3).astype(int)
        return combined_pred


def ensemble_full(input_path='data/api_simulation_data.csv', output_dir='outputs'):
    print("🔹 Loading models...")
    models = load_all_models()

    print(f"🔹 Reading input data from {input_path}...")
    data = pd.read_csv(input_path)

    if 'Class' in data.columns:
        y_true = data['Class']
        data = data.drop('Class', axis=1)
    else:
        y_true = None

    # Make predictions with majority voting ensemble
    ensemble_model = MajorityVotingEnsemble(models)
    combined_pred = ensemble_model.predict(data)

    print(f"✅ Ensemble Full predictions completed. Sample output:\n{combined_pred[:10]}")

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({'ensemble_prediction': combined_pred}).to_csv(
        os.path.join(output_dir, 'ensemble_full_predictions.csv'),
        index=False
    )

    # Save the model for reuse/evaluation
    model_path = os.path.join(output_dir, 'ensemble_full_model.joblib')
    dump(ensemble_model, model_path)
    print(f"✅ Ensemble model saved to {model_path}")

    if y_true is not None:
        print("🔹 Calculating evaluation metrics...")
        report = classification_report(y_true, combined_pred, digits=4)
        cm = confusion_matrix(y_true, combined_pred)
        fpr, tpr, _ = roc_curve(y_true, combined_pred)
        roc_auc = auc(fpr, tpr)

        print(report)
        print(f"ROC AUC: {roc_auc:.4f}")

        with open(os.path.join(output_dir, 'ensemble_full_report.txt'), 'w') as f:
            f.write(report)
            f.write(f"\nROC AUC: {roc_auc:.4f}")

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Full - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, 'ensemble_full_confusion_matrix.png'))
        plt.close()

        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble Full - ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'ensemble_full_roc_curve.png'))
        plt.close()

        print(f"✅ Metrics saved in '{output_dir}'")

    return combined_pred


if __name__ == "__main__":
    ensemble_full()
