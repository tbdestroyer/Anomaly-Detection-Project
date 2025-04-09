FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and scripts
COPY isolation_forest_model.joblib .
COPY evaluate_model.py .
COPY creditcard_test.csv .

# Create a directory for outputs
RUN mkdir -p /app/outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the evaluation script when container starts
CMD ["python", "evaluate_model.py"] 