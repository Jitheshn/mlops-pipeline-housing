# src/train.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate_model(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Log params and metrics
        mlflow.log_param("model_type", model_name)
        metrics = evaluate_model(y_test, predictions)
        mlflow.log_metrics(metrics)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Return R2 and run ID for comparison
        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "r2": metrics["r2"]
        }


def main():
    df = pd.read_csv("data/raw/housing.csv")
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5)
    }

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("CaliforniaHousingExperiment")

    best_result = {"r2": -np.inf}

    for name, model in models.items():
        result = train_and_log_model(
            model, name, X_train, X_test, y_train, y_test)
        if result["r2"] > best_result["r2"]:
            best_result = result

    # Register best model
    model_uri = f"runs:/{best_result['run_id']}/model"
    mlflow.register_model(model_uri, "CalHousingBestModel")

    print(
        f"✅ Best model ({
            best_result['model_name']}) registered with R² = {
            best_result['r2']:.4f}")

    # ✅ Save best model locally to models/best_model.pkl
    print("🔁 Downloading best model from MLflow...")
    model = mlflow.sklearn.load_model(model_uri)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    print("✅ Best model saved to models/best_model.pkl")


if __name__ == "__main__":
    main()
