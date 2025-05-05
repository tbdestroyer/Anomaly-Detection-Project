
import pandas as pd
import time
import csv
import os
from locust import HttpUser, task, between, events

LOG_FILE = "prediction_results_log.csv"
REQUESTS_CSV = "logs/requests.csv"

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch_id", "y_true", "y_pred", "latency_ms"])
    with open(REQUESTS_CSV, mode='w') as f:
        f.write("")
    print("🔄 Logs reset. Ready for new test!")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.entries.get(("/predict", "POST"))
    header = "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,50%,75%,95%,99%\n"
    with open(REQUESTS_CSV, "w") as f:
        f.write(header)
        f.write(f"POST,/predict,{stats.num_requests},{stats.num_failures},{stats.median_response_time},{stats.avg_response_time},{stats.min_response_time},{stats.max_response_time},{stats.avg_content_length},{stats.total_rps},{stats.total_fail_per_sec},{stats.get_response_time_percentile(50)},{stats.get_response_time_percentile(75)},{stats.get_response_time_percentile(95)},{stats.get_response_time_percentile(99)}\n")
    print("✅ Test completed. requests.csv saved!")

class AnomalyTestUser(HttpUser):
    wait_time = between(0.5, 1)
    host = "http://127.0.0.1:8000"

    def on_start(self):
        full_data = pd.read_csv('data/api_simulation_data.csv')
        self.labels = full_data['Class'].values
        self.features = full_data.drop(['Class'], axis=1).values
        self.batch_size = 3
        self.current_index = 0

    @task
    def send_prediction_request(self):
        if self.current_index >= 50000:
            self.environment.runner.quit()
            return
        batch_data = self.features[self.current_index:self.current_index + self.batch_size].tolist()
        batch_labels = self.labels[self.current_index:self.current_index + self.batch_size].tolist()
        self.current_index += self.batch_size
        payload = {"data": batch_data}
        start_time = time.time()
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            latency = (time.time() - start_time) * 1000
            if response.status_code == 200:
                preds = response.json().get("predictions", [])
                with open(LOG_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.current_index // self.batch_size, batch_labels, preds, round(latency, 2)])
            else:
                response.failure(f"Failed with status {response.status_code}")
