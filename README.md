# Anomaly-Detection-Project
A cloud-based anomaly detection system using MLOps

# Credit Card Fraud Detection - Isolation Forest Model

This project contains a containerized Isolation Forest model for credit card fraud detection.

## Files Included

- `isolation_forest_model.joblib`: The trained Isolation Forest model
- `evaluate_model.py`: Script to evaluate the model performance
- `creditcard_test.csv`: Test dataset for evaluation 
## https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download
- `Dockerfile`: Instructions for building the Docker container
- `requirements.txt`: Python dependencies
- `.dockerignore`: Files to exclude from Docker build

## Building the Docker Image

To build the Docker image, run:

```bash
docker build -t fraud-detection-model .
```

## Running the Container

To run the container and evaluate the model:

```bash
docker run -v $(pwd)/outputs:/app/outputs fraud-detection-model
```

This will:
1. Run the evaluation script
2. Generate performance metrics
3. Create visualizations
4. Save outputs to the `outputs` directory

## Outputs

The container will generate:
- `model_performance_report.csv`: Detailed performance metrics
- `model_evaluation_curves.png`: ROC and Precision-Recall curves

## Requirements

- Docker
- At least 2GB of RAM
- 1GB of disk space
