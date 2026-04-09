from src.models.predict import load_model, predict
import joblib

# Load model and scaler
model = load_model("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Sample input (23 features)
sample_input = [
    20000, 2, 2, 1, 24,
    -1, -1, -1, -1, -1, -1,
    3913, 3102, 689, 0, 0, 0,
    0, 689, 0, 0, 0, 0
]

result = predict(model, scaler, sample_input)

print(result)