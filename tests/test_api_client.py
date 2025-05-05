import requests
import pandas as pd
import json

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Load a few sample rows from your api_simulation_data.csv
data = pd.read_csv('api_simulation_data.csv')

# Drop 'Class' column since the API expects only features
features = data.drop('Class', axis=1)

# Select first 5 rows for testing
sample_data = features.head(5).values.tolist()

# Prepare payload
payload = {
    "data": sample_data
}

# Send POST request
response = requests.post(url, json=payload)

# Display response
if response.status_code == 200:
    print("✅ Response from API:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"❌ Failed with status code {response.status_code}")
    print(response.text)
