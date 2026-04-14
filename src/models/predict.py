import joblib
import numpy as np
import pandas as pd

def load_model(model_path: str):
    return joblib.load(model_path)


def predict(model, scaler, input_data):

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data], columns=scaler.feature_names_in_)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict probability
    prob = model.predict_proba(input_scaled)[0][1]

    # 🔥 Threshold (you can tune this later)
    THRESHOLD = 0.30

    prediction = 1 if prob > THRESHOLD else 0

    # 🔥 Better aligned risk scoring
    if prob < 0.3:
        risk = "Low"
    elif prob < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "prediction": int(prediction),
        "probability": round(float(prob), 4),
        "risk_level": risk
    }