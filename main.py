import joblib
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.train import train_models

def main():
    df = load_data("data/raw/credit_default.csv")

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    best_model, results = train_models(X_train, y_train, X_test, y_test)
    joblib.dump(scaler, "models/scaler.pkl")

    print("\nFinal Results:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()