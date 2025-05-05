import pandas as pd

# Correct variable name and use raw string for the file path
creditcard_csv_path = r"C:\Users\tanbu\Documents\GitHub\Anomaly-Detection-Project\creditcard.csv"

# Load the dataset using the variable
data = pd.read_csv(creditcard_csv_path)

# Display the first few rows
print(data.head())
