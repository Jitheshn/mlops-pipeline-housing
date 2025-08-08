from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import logging
import sqlite3
from datetime import datetime
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI()

# ====== Load Model ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
model = joblib.load(MODEL_PATH)

# ====== Configure Logging to File ======
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "predictions.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ====== SQLite Setup ======
DB_PATH = os.path.join(LOG_DIR, "predictions.db")
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


# ====== Prometheus Metric ======
prediction_counter = Counter("prediction_requests_total", "Total number of prediction requests")


# ====== Request Schema ======
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# ====== Routes ======
@app.get("/")
def read_root():
    return {"message": "Welcome to California Housing Prediction API"}


@app.post("/predict")
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
