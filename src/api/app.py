from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from src.models.predict import load_model, predict

app = FastAPI()

# Load once at startup
model = load_model("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}

@app.post("/predict")
def predict_endpoint(data: InputData):
    return predict(model, scaler, data.features)