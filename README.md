# MLOps (S2-24_AIMLCZG523) Assignment -1 
# 🏠 Housing Price Prediction API (MLOps Pipeline)

This project implements a complete MLOps pipeline for predicting California housing prices using machine learning. It includes model training, MLflow tracking and model registry, and a FastAPI-powered REST API for serving the best model. The solution is containerized using Docker for easy deployment.

Group No: 7
 
Group Member Names:
1. Mrs. VishnuPriya R | 2023ac05678@wilp.bits-pilani.ac.in 
2. Mr. Ravichander R |  2023ac05152@wilp.bits-pilani.ac.in 
3. Mr. Jithesh Nair |  2023ac05661@wilp.bits-pilani.ac.in 
4. Mrs. Priyanka Bhambure | 2023ac05792@wilp.bits-pilani.ac.in  


---

## 📂 Project Structure

```
mlops-pipeline-housing/
│
├── data/
│   └── raw/
│       └── housing.csv           # Dataset (California housing)
│
├── src/
│   └── train.py                  # ML model training and logging to MLflow
│
├── app.py                        # FastAPI app serving the best model
├── Dockerfile                    # Docker container definition
├── requirements.txt              # Required Python packages
├── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8+
- pip
- Docker (for containerization)
- Git (optional)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mlops-pipeline-housing.git
cd mlops-pipeline-housing
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

To train the model and log experiments to MLflow:

```bash
python src/train.py
```

- It trains a Linear Regression and a Decision Tree model.
- Automatically selects the best model using R² score.
- Registers the best model to the MLflow Model Registry as `CalHousingBestModel`.

---

## 🚀 Run the API Locally

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

Once running, access the API docs at:  
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔮 API Endpoint

### `POST /predict`

Predict median house value for a given input:

#### ✅ Sample Request:

```json
POST /predict
Content-Type: application/json

{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

#### ✅ Sample Response:

```json
{
  "prediction": 4.534
}
```

---

## 🐳 Docker Usage

### Build Docker Image

```bash
docker build -t housing-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 housing-api
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing

You can test the `/predict` endpoint using:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Postman
- `curl` or any HTTP client

---

## 📈 MLflow Integration

The project uses MLflow to:

- Track experiments and metrics
- Log and register models
- Load the best model for prediction

You can launch the MLflow UI using:

```bash
mlflow ui
```

Then visit: [http://localhost:5000](http://localhost:5000)

---

## ✨ Enhancements (Future Scope)

- CI/CD with GitHub Actions
- Model monitoring
- Unit tests with pytest
- Add support for hyperparameter tuning
- Deploy to cloud (Azure/AWS/GCP)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

- [scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/)
- [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
