import joblib
import numpy as np
import pandas as pd

def load_model(model_path: str):
    """
    Load trained model from disk
    """
    return joblib.load(model_path)


def predict(model, scaler, input_data):
    """
    Predict default risk
    """

    # Convert input to DataFrame with correct column names
    input_df = pd.DataFrame([input_data], columns=scaler.feature_names_in_)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict probability
    prob = model.predict_proba(input_scaled)[0][1]

    # Class prediction
    prediction = 1 if prob > 0.35 else 0

    # Risk scoring
    if prob < 0.3:
        risk = "Low"
    elif prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "prediction": int(prediction),
        "probability": float(f"{prob:.4f}"),
        "risk_level": risk
    }