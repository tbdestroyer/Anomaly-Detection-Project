FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p logs outputs outputs/drift_plots models

# Copy all necessary files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose ports for FastAPI, Locust, and Streamlit
EXPOSE 8000 8089 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting Anomaly Detection Project..."\n\
\n\
# Step 1: Data Preparation\n\
echo "Preparing data..."\n\
if [ ! -f "creditcard.csv" ]; then\n\
    echo "Error: creditcard.csv not found. Please make sure the data file is present."\n\
    exit 1\n\
fi\n\
\n\
echo "Preprocessing credit card data..."\n\
python creditcard_preprocess.py || { echo "Error preprocessing data"; exit 1; }\n\
\n\
echo "Scaling features..."\n\
python feature_scaling.py || { echo "Error scaling features"; exit 1; }\n\
\n\
echo "Splitting data into training and API sets..."\n\
python data_split.py || { echo "Error splitting data"; exit 1; }\n\
\n\
echo "Analyzing data and generating visualizations..."\n\
python data_analysis.py || { echo "Error in data analysis"; exit 1; }\n\
\n\
# Step 2: Train individual models\n\
echo "Training individual models..."\n\
\n\
echo "Training Isolation Forest..."\n\
python models/train_isolation_forest.py || { echo "Error training Isolation Forest"; exit 1; }\n\
\n\
echo "Training One-Class SVM..."\n\
python models/train_oneclass_svm.py || { echo "Error training One-Class SVM"; exit 1; }\n\
\n\
echo "Training Elliptic Envelope..."\n\
python models/train_elliptic_envelope.py || { echo "Error training Elliptic Envelope"; exit 1; }\n\
\n\
echo "Training Local Outlier Factor..."\n\
python models/train_lof.py || { echo "Error training Local Outlier Factor"; exit 1; }\n\
\n\
echo "Training Autoencoder..."\n\
python models/train_autoencoder.py || { echo "Error training Autoencoder"; exit 1; }\n\
\n\
# Step 3: Evaluate models and generate reports\n\
echo "Evaluating models and generating reports..."\n\
python evaluate_model.py || { echo "Error evaluating models"; exit 1; }\n\
\n\
echo "Generating benchmark report..."\n\
python generate_benchmark_report.py || { echo "Error generating benchmark report"; exit 1; }\n\
\n\
echo "Generating full report..."\n\
python generate_full_report.py || { echo "Error generating full report"; exit 1; }\n\
\n\
# Step 4: Train ensemble models\n\
echo "Training ensemble models..."\n\
\n\
echo "Training Light Ensemble..."\n\
python models/ensemble_light.py || { echo "Error training Light Ensemble"; exit 1; }\n\
\n\
echo "Training Medium Ensemble..."\n\
python models/ensemble_medium.py || { echo "Error training Medium Ensemble"; exit 1; }\n\
\n\
echo "Training Full Ensemble..."\n\
python models/ensemble_full.py || { echo "Error training Full Ensemble"; exit 1; }\n\
\n\
# Step 5: Run drift detection\n\
echo "Running drift detection..."\n\
python drift_detection.py || { echo "Warning: Drift detection completed with issues"; }\n\
\n\
# Step 6: Start FastAPI server\n\
echo "Starting FastAPI server..."\n\
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug &\n\
\n\
# Wait for FastAPI server to start and models to load\n\
echo "Waiting for models to load (this may take a few minutes)..."\n\
sleep 30\n\
\n\
# Check if FastAPI is responding and models are loaded\n\
echo "Checking if FastAPI server is ready..."\n\
until curl -s http://localhost:8000/ > /dev/null; do\n\
    echo "Waiting for FastAPI server to start..."\n\
    sleep 5\n\
done\n\
\n\
# Test API prediction endpoint\n\
echo "Testing API prediction endpoint..."\n\
curl -s -X POST http://localhost:8000/predict \\\n\
    -H "Content-Type: application/json" \\\n\
    -d "{\"data\":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}" \\\n\
    || { echo "Error: API prediction endpoint is not working correctly"; exit 1; }\n\
echo "FastAPI server is ready and working correctly!"\n\
\n\
# Step 7: Start Locust\n\
echo "Starting Locust load testing..."\n\
echo "Verifying API is ready for Locust..."\n\
curl -s http://localhost:8000/ > /dev/null || { echo "Error: API is not ready for Locust testing"; exit 1; }\n\
\n\
echo "Starting Locust with web UI..."\n\
locust -f locustfile.py --host http://localhost:8000 --web-host 0.0.0.0 &\n\
\n\
# Wait for Locust to start\n\
sleep 5\n\
\n\
# Verify Locust is running\n\
echo "Checking if Locust is running..."\n\
curl -s http://localhost:8089/ > /dev/null || echo "Warning: Locust web interface is not accessible. Continuing anyway..."\n\
\n\
# Step 8: Start Streamlit dashboard\n\
echo "Starting Streamlit dashboard..."\n\
streamlit run evaluation_dashboard.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Set the startup script as the entrypoint
ENTRYPOINT ["/app/start.sh"]
