import streamlit as st
import requests

st.title("Loan Default Risk Predictor")

st.write("Enter customer financial details:")

st.set_page_config(page_title="Loan Risk Predictor", layout="centered")
LIMIT_BAL = st.number_input("Credit Limit", value=20000)
AGE = st.number_input("Age", value=24)

PAY_0 = st.selectbox("Recent Payment Status", [-2, -1, 0, 1, 2, 3])
BILL_AMT1 = st.number_input("Last Bill Amount", value=3000)
PAY_AMT1 = st.number_input("Last Payment Amount", value=1000)

EDUCATION = st.selectbox("Education Level", [1, 2, 3, 4])
MARRIAGE = st.selectbox("Marital Status", [1, 2, 3])

if st.button("Predict Risk"):

    features = [
        LIMIT_BAL, 2, EDUCATION, MARRIAGE, AGE,
        PAY_0, 0, 0, 0, 0, 0,
        BILL_AMT1, 0, 0, 0, 0, 0,
        PAY_AMT1, 0, 0, 0, 0, 0
    ]

    try:
        response = requests.post(
            "https://loan-default-api-mqxj.onrender.com/predict",
            json={"features": features}
        )
        result = response.json()

        st.subheader("Prediction Result")
        risk = result["risk_level"]
        prob = result["probability"]
        if risk == "High":
            st.error("High Risk of Default")
        elif risk == "Medium":
            st.warning("Moderate Risk")
        else:
            st.success("Low Risk Customer")

        st.metric("Default Probability", prob)

        st.write("---")
    except:
        st.error("API not running. Start FastAPI server.")