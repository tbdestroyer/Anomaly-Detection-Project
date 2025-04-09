import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def scale_features(file_path):
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Identify numerical columns (excluding the target variable 'Class' if it exists)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if 'Class' in numerical_cols:
        numerical_cols = numerical_cols[numerical_cols != 'Class']
    
    # Create a copy of the dataframe to store scaled features
    df_scaled = df.copy()
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Scale the numerical features
    print("\nScaling numerical features...")
    df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Display basic statistics of scaled features
    print("\n=== Statistics of Scaled Features ===")
    print(df_scaled[numerical_cols].describe())
    
    # Save the scaled dataset
    output_path = 'creditcard_scaled.csv'
    df_scaled.to_csv(output_path, index=False)
    print(f"\nScaled dataset saved as '{output_path}'")
    
    return df_scaled

if __name__ == "__main__":
    # Use the same path as in the data analysis script
    dataset_path = r"C:\Users\tanbu\Documents\GitHub\Anomaly-Detection-Project\creditcard.csv"
    scaled_data = scale_features(dataset_path) 