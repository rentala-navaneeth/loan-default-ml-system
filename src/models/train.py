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


# 🔥 Global threshold (same as predict.py)
THRESHOLD = 0.30


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using threshold-based prediction
    """

    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # 🔥 Apply threshold instead of default 0.5
    y_pred = (y_prob > THRESHOLD).astype(int)

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

    # 🔥 Handle class imbalance for XGBoost
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )
    }

    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Cross-validation
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc"
        )
        print(f"CV ROC-AUC: {round(np.mean(cv_scores), 4)}")

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Round metrics
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        print("Test Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

        results[name] = metrics

        if metrics["F1"] > best_score:
            best_score = metrics["F1"]
            best_model = model

        print("\nThreshold vs Recall:")
        probs = model.predict_proba(X_test)[:, 1]
        for t in [0.2, 0.25, 0.3, 0.35, 0.4]:
            preds = (probs > t).astype(int)
            recall = recall_score(y_test, preds)
            print(f"  Threshold {t}: Recall {round(recall, 4)}")

    print("\nBest Model Selected based on F1 Score")

    # Save best model
    joblib.dump(best_model, "models/best_model.pkl")

    return best_model, results