from locust import HttpUser, task, between
import pandas as pd
import json

class AnomalyDetectionUser(HttpUser):
    wait_time = between(0.5, 1)  # Simulate user wait time between requests

    def on_start(self):
        # Load sample data once per simulated user
        data = pd.read_csv('api_simulation_data.csv').drop('Class', axis=1)
        self.sample_payload = {
            "data": data.head(3).values.tolist()   # Send 3 rows per request
        }

    @task
    def predict_anomalies(self):
        self.client.post("/predict", json=self.sample_payload)
