🚀 Loan Default Risk Prediction System

An end-to-end machine learning system to predict the likelihood of a customer defaulting on credit, built with a production-style architecture.

📌 Features
Trained multiple models (Logistic Regression, Random Forest, XGBoost)
Selected best model based on ROC-AUC
Implemented class imbalance handling and threshold tuning
Built FastAPI backend for real-time inference
Developed Streamlit UI for user interaction
Deployed API on cloud
🧠 Model Performance (XGBoost)
Metric	Score
Accuracy	0.8118
Precision	0.6289
Recall	0.3640
F1 Score	0.4611
ROC-AUC	0.7565
🏗️ System Architecture
User → Streamlit UI → FastAPI → ML Model → Response
🖥️ Run Locally
1. Install dependencies
pip install -r requirements.txt
2. Run API
uvicorn src.api.app:app --reload
3. Run UI
streamlit run app_ui.py
🌍 Live API

👉 https://loan-default-api-mqxj.onrender.com/docs

📂 Project Structure

loan-default-ml-system/
│
├── data/
│   └── raw/
│       └── credit_default.csv
│
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│
├── app_ui.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

🎯 Use Case

Simulates a real-world credit risk assessment system where financial inputs are used to predict default risk in real-time.