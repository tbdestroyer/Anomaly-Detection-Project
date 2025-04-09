import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_isolation_forest(train_path, test_path, contamination=0.01, random_state=42):
    # Load the training and testing datasets
    print("Loading datasets...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']
    
    # Initialize and train the Isolation Forest model
    print("\nTraining Isolation Forest model...")
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the model
    iso_forest.fit(X_train)
    
    # Make predictions
    # Note: IsolationForest returns -1 for anomalies and 1 for normal points
    # We need to convert these to match our original labels (0 for normal, 1 for fraud)
    y_pred_train = iso_forest.predict(X_train)
    y_pred_test = iso_forest.predict(X_test)
    
    # Convert predictions to match original labels
    y_pred_train = np.where(y_pred_train == 1, 0, 1)
    y_pred_test = np.where(y_pred_test == 1, 0, 1)
    
    # Print model performance
    print("\n=== Model Performance ===")
    print("\nTraining Set Performance:")
    print(classification_report(y_train, y_pred_train))
    print("\nTesting Set Performance:")
    print(classification_report(y_test, y_pred_test))
    
    # Create and save confusion matrices
    plt.figure(figsize=(12, 5))
    
    # Training set confusion matrix
    plt.subplot(1, 2, 1)
    cm_train = confusion_matrix(y_train, y_pred_train)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
    plt.title('Training Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Testing set confusion matrix
    plt.subplot(1, 2, 2)
    cm_test = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
    plt.title('Testing Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    
    # Save the trained model
    model_path = 'isolation_forest_model.joblib'
    joblib.dump(iso_forest, model_path)
    print(f"\nModel saved as '{model_path}'")
    
    return iso_forest

if __name__ == "__main__":
    train_path = 'creditcard_train.csv'
    test_path = 'creditcard_test.csv'
    model = train_isolation_forest(train_path, test_path) 