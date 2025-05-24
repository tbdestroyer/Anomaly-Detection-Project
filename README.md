# Scalable Anomaly Detection with MLOps and HPC Integration

**Team Members**: Haider Khan, Ali Fehmi Yildiz, Taner Baki Bulbul  
**Course**: MSML 605  
**Advisor**: Dr. Samet Ayhan  
**Git**: [Anomaly-Detection-Project](https://github.com/tbdestroyer/Anomaly-Detection-Project)

**Note**: Click here to download the final report for this project.
[final_report_605.pdf](https://github.com/user-attachments/files/20421028/final_report_605.pdf)

## Abstract

In this project, we develop a scalable anomaly detection system for credit card fraud, integrating Machine Learning Operations (MLOps) practices and High-Performance Computing (HPC). We address the detection of fraudulent transactions as an anomaly detection problem using an ensemble of varied algorithms (including One-Class SVM, Isolation Forest, Autoencoder, Local Outlier Factor, and Elliptic Envelope) to improve detection robustness.

Our solution incorporates ClearML for experiment tracking and orchestration, Docker containers for consistent deployment, the Zaratan HPC cluster for accelerated training and inference, and Locust for stress-testing the deployed model API. We applied our pipeline to the highly imbalanced Kaggle Credit Card Fraud Dataset, achieving significant improvements in detection performance and scalability. The ensemble model, tuned via ClearML on HPC resources, attained higher fraud detection accuracy (macro-F1 ~97.5%) than a baseline one-class SVM (~95%) while maintaining low false-positive rates. The deployed system demonstrated the capability to handle heavy load (~220 requests/second) with low latency (~80 ms average) in testing.

This report details the problem context and related work, our proposed solution architecture, evaluation results with comprehensive metrics, and discusses current limitations (e.g., inference latency, lack of drift detection, limited cloud integration) along with future improvements. The integration of MLOps and HPC proved effective in enhancing both the performance and scalability of anomaly/fraud detection for real-world use.

---





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
