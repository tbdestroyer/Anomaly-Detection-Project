import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_data(file_path, test_size=0.2, random_state=42):
    # Load the scaled dataset
    print("Loading scaled dataset...")
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop('Class', axis=1)  # Assuming 'Class' is the target column
    y = df['Class']
    
    # Split the data
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y  # This ensures the same proportion of samples for each class
    )
    
    # Create DataFrames for the split data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save the split datasets
    train_path = 'creditcard_train.csv'
    test_path = 'creditcard_test.csv'
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    # Print information about the split
    print("\n=== Dataset Split Information ===")
    print(f"Total samples: {len(df)}")
    print(f"Training set size: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
    print(f"Testing set size: {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")
    
    # Print class distribution in both sets
    print("\n=== Class Distribution ===")
    print("Training set:")
    print(train_data['Class'].value_counts(normalize=True))
    print("\nTesting set:")
    print(test_data['Class'].value_counts(normalize=True))
    
    return train_data, test_data

if __name__ == "__main__":
    # Use the scaled dataset
    scaled_data_path = 'creditcard_scaled.csv'
    train_data, test_data = split_data(scaled_data_path) 