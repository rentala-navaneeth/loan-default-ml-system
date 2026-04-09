import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from xgboost import XGBClassifier


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    return metrics


def train_models(X_train, y_train, X_test, y_test):
    """
    Train and compare multiple models
    """

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"CV ROC-AUC: {round(np.mean(cv_scores), 4)}")

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Round metrics once
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        print("Test Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

        results[name] = metrics

        # Select best model based on ROC-AUC
        if metrics["ROC-AUC"] > best_score:
            best_score = metrics["ROC-AUC"]
            best_model = model

    print("\nBest Model Selected based on ROC-AUC")

    # Save best model
    joblib.dump(best_model, "models/best_model.pkl")

    return best_model, results