from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import logging
import sqlite3
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import subprocess

app = FastAPI()

# ====== Paths ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(LOG_DIR, "predictions.db")

# ====== Load Model ======
model = joblib.load(MODEL_PATH)

# ====== Logging Setup ======
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "predictions.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ====== SQLite Setup ======
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_logs (
    timestamp TEXT,
    features TEXT,
    prediction REAL
)
""")
conn.commit()

# ====== Prometheus Metrics ======
prediction_counter = Counter("prediction_requests_total", "Total number of prediction requests")
prediction_latency = Histogram("prediction_latency_seconds", "Time taken for prediction requests")


# ====== Input Schema with Validation ======
class InputData(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income in block group (must be positive)")
    HouseAge: float = Field(..., ge=1, le=100, description="Median house age in years (1-100)")
    AveRooms: float = Field(..., gt=0, description="Average number of rooms per household")
    AveBedrms: float = Field(..., gt=0, description="Average number of bedrooms per household")
    Population: float = Field(..., ge=1, description="Block group population (must be >= 1)")
    AveOccup: float = Field(..., gt=0, description="Average occupants per household")
    Latitude: float = Field(..., ge=32, le=42, description="Latitude in California range")
    Longitude: float = Field(..., ge=-125, le=-113, description="Longitude in California range")


# ====== Routes ======
@app.get("/")
def read_root():
    return {"message": "Welcome to California Housing Prediction API"}


@app.post("/predict")
@prediction_latency.time()
def predict(data: InputData):
    prediction_counter.inc()  # Increment metric counter

    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    prediction_value = float(prediction[0])

    # Log to file
    logging.info(f"Input: {data.dict()} -> Prediction: {prediction_value}")

    # Log to SQLite
    cursor.execute(
        "INSERT INTO prediction_logs VALUES (?, ?, ?)",
        (datetime.now().isoformat(), str(data.dict()), prediction_value)
    )
    conn.commit()
    return {"prediction": prediction_value}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/retrain")
def retrain_model():
    """
    Retrains the model using new data and reloads it.
    Assumes there is a training script at src/train.py
    """
    training_script = os.path.join(BASE_DIR, "src", "train.py")
    if not os.path.exists(training_script):
        return {"error": f"Training script not found at {training_script}"}

    try:
        subprocess.run(["python", training_script], check=True)
        global model
        model = joblib.load(MODEL_PATH)
        return {"status": "Model retrained and reloaded successfully"}
    except subprocess.CalledProcessError as e:
        return {"error": f"Retraining failed: {str(e)}"}
