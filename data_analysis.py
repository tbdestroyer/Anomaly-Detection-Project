import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def analyze_dataset(file_path="creditcard.csv"):
    # Validate file path
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    # Load the dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Basic information about the dataset
    print("\n=== Dataset Information ===")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    
    # Check for missing values
    print("\n=== Missing Values Analysis ===")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    print(missing_info[missing_info['Missing Values'] > 0])
    
    # Check for duplicates
    print("\n=== Duplicate Analysis ===")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    print(f"Percentage of duplicates: {(duplicates/len(df))*100:.2f}%")
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Distribution of numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        plt.figure(figsize=(15, 5 * len(numerical_cols)))
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(len(numerical_cols), 1, i)
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig('numerical_distributions.png')
        print("Saved numerical distributions as 'numerical_distributions.png'")
    else:
        print("No numerical columns found for distribution plots.")
    
    # Correlation heatmap
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        print("Saved correlation heatmap as 'correlation_heatmap.png'")
    else:
        print("Not enough numerical columns for a correlation heatmap.")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    analyze_dataset()