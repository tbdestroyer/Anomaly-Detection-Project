FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libxft-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY isolation_forest_model.joblib .
COPY evaluate_model.py .
COPY creditcard_test.csv .

# Create output directory
RUN mkdir -p /app/outputs

ENV PYTHONUNBUFFERED=1

# Set default command with environment parameter
CMD ["python", "evaluate_model.py", "--env", "Docker (Local)"]
