from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import time

from models.load_models import load_all_models
import warnings
import logging

logging.getLogger("uvicorn.access").disabled = True

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="Anomaly Detection API", version="1.0")

# Load models once at startup
models = {}

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    print("🔹 Loading ensemble_medium models...")
    models = load_all_models()
    print("✅ Models loaded successfully!")
    yield  # This allows the app to continue running after setup
    # You could add cleanup logic here if needed when app shuts down

app = FastAPI(title="Anomaly Detection API", version="1.0", lifespan=lifespan)


# Define input schema
class DataInput(BaseModel):
    data: list  # Expecting list of lists (rows of features)

def predict_autoencoder(ae_model, ae_threshold, X):
    reconstructed = ae_model.predict(X, verbose=0)

    mse = np.mean(np.square(X - reconstructed), axis=1)
    return (mse > ae_threshold).astype(int)

@app.post("/predict")
def predict(input_data: DataInput):
    try:
        X = pd.DataFrame(input_data.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")

    start_time = time.time()

    # Individual predictions
    svm_pred = np.where(models['svm'].predict(X) == 1, 0, 1)
    elliptic_pred = np.where(models['elliptic'].predict(X) == 1, 0, 1)
    ae_pred = predict_autoencoder(models['autoencoder'], models['ae_threshold'], X)

    # Majority voting (Threshold = 2 for ensemble_medium)
    combined_pred = ((svm_pred + elliptic_pred + ae_pred) >= 2).astype(int)

    end_time = time.time()
    latency = round((end_time - start_time) * 1000, 2)  # in milliseconds

    return {
        "predictions": combined_pred.tolist(),
        "latency_ms": latency
    }

@app.get("/")
def read_root():
    return {"message": "Anomaly Detection API is running."}
