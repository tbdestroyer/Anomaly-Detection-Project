# Anomaly Detection System

## Architecture Overview
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Data Pipeline  │────▶│  Model Training │────▶│  Model Serving  │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Drift Detection│     │  Model Ensemble │     │  Load Testing   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Components

### 1. Data Pipeline
- `creditcard_preprocess.py`: Data preprocessing
- `feature_scaling.py`: Feature standardization
- `data_split.py`: Train/test split
- `drift_detection.py`: Data drift monitoring

### 2. Model Training
- Individual Models:
  - `train_isolation_forest.py`
  - `train_oneclass_svm.py`
  - `train_elliptic_envelope.py`
  - `train_autoencoder.py`
- Ensemble Models:
  - `ensemble_light.py`
  - `ensemble_medium.py`
  - `ensemble_full.py`

### 3. Model Serving
- `main.py`: FastAPI server
- `load_models.py`: Model loading and prediction
- `locustfile.py`: Load testing configuration

### 4. Monitoring & Evaluation
- `evaluate_model.py`: Model performance evaluation
- `generate_benchmark_report.py`: Performance metrics
- `generate_full_report.py`: Comprehensive analysis
- `evaluation_dashboard.py`: Streamlit visualization

## Quick Start with Docker (No Python or manual setup needed!)

1. **Install Docker Desktop:**  
   [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

2. **Download this project and place `creditcard.csv` in the project folder.**

3. **Open a terminal in the project directory and run:**
   ```sh
   docker build -t anomaly-detection .
   ```
   Then:
   ```sh
   # For Windows CMD
   docker run -p 8000:8000 -p 8089:8089 -p 8501:8501 -v %cd%/creditcard.csv:/app/creditcard.csv anomaly-detection

   # For PowerShell or Mac/Linux
   docker run -p 8000:8000 -p 8089:8089 -p 8501:8501 -v ${PWD}/creditcard.csv:/app/creditcard.csv anomaly-detection
   ```

4. **Open these in your browser:**
   - [http://localhost:8000](http://localhost:8000) (API)
   - [http://localhost:8089](http://localhost:8089) (Locust)
   - [http://localhost:8501](http://localhost:8501) (Dashboard)

---

## Setup Instructions

### Option 1: Docker Setup (Recommended)
1. Install Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Build the Docker image:
```bash
docker build -t anomaly-detection .
```
3. Run the container:
```bash
docker run -p 8000:8000 -p 8089:8089 -p 8501:8501 -v ${PWD}/creditcard.csv:/app/creditcard.csv anomaly-detection
```
4. Access the services:
   - FastAPI: http://localhost:8000
   - Locust: http://localhost:8089
   - Streamlit Dashboard: http://localhost:8501

### Option 2: Manual Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete pipeline:
```bash
.\run_project.bat
```

3. For HPC deployment:
```bash
sbatch zaratan_train.sh  # For training
sbatch zaratan_infer.sh  # For inference
```

## API Usage

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Make predictions:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}'
```

## Monitoring

1. Drift Detection:
```bash
python drift_detection.py
```

2. Load Testing:
```bash
locust -f locustfile.py --host http://localhost:8000
```

3. Dashboard:
```bash
streamlit run evaluation_dashboard.py
```

## Performance Metrics
- Precision: 0.14
- Recall: 0.98
- F1 Score: 0.24
- ROC-AUC: 0.98

## License
MIT License
