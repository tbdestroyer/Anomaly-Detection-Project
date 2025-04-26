import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def build_autoencoder(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(train_path='creditcard_train.csv', output_dir='outputs'):
    print("🔹 Loading pre-scaled training data...")
    data = pd.read_csv(train_path)
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Train only on normal transactions
    X_train = X[y == 0]

    print(f"🔹 Training AutoEncoder on {X_train.shape[0]} normal transactions...")
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)

    history = autoencoder.fit(
        X_train, X_train,
        epochs=20,
        batch_size=256,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )

    print("✅ Training completed! Calculating reconstruction errors...")
    reconstructions = autoencoder.predict(X)
    mse = np.mean(np.square(X - reconstructions), axis=1)

    # Set threshold (95th percentile of normal reconstruction errors)
    threshold = np.percentile(mse[y == 0], 95)
    print(f"🔹 Anomaly detection threshold set at: {threshold:.6f}")

    # Predict anomalies
    y_pred = (mse > threshold).astype(int)

    print("🔹 Calculating evaluation metrics...")
    report = classification_report(y, y_pred, digits=4)
    cm = confusion_matrix(y, y_pred)
    fpr, tpr, _ = roc_curve(y, mse)
    roc_auc = auc(fpr, tpr)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save classification report
    with open(os.path.join(output_dir, 'autoencoder_classification_report.txt'), 'w') as f:
        f.write(report)
        f.write(f"\nROC AUC: {roc_auc:.4f}")
        f.write(f"\nThreshold: {threshold:.6f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('AutoEncoder - Confusion Matrix (Train Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'autoencoder_confusion_matrix.png'))
    plt.close()

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AutoEncoder - ROC Curve (Train Set)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'autoencoder_roc_curve.png'))
    plt.close()

    # Save model and threshold
    autoencoder.save(os.path.join(output_dir, 'autoencoder_model.h5'))
    joblib.dump(threshold, os.path.join(output_dir, 'autoencoder_threshold.joblib'))

    print(f"✅ AutoEncoder model, threshold, and metrics saved in '{output_dir}'")

if __name__ == "__main__":
    train_autoencoder()
