import pandas as pd

# Use the correct path for the current directory
creditcard_csv_path = "creditcard.csv"

# Load the dataset using the variable
data = pd.read_csv(creditcard_csv_path)

# Display the first few rows
print(data.head())
