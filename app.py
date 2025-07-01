import streamlit as st
import pandas as pd
import joblib

# Load model dan threshold
model = joblib.load("stacking_model.pkl")
threshold = joblib.load("optimal_threshold.pkl")

st.title("Customer Churn Prediction")
st.write("Masukkan data pelanggan di bawah ini untuk memprediksi kemungkinan churn.")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 3000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Buat DataFrame dari input
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    "InternetService": [internet_service]
    
})

if st.button("Prediksi Churn"):
    prob = model.predict_proba(input_data)[0][1]
    pred = int(prob >= threshold)

    st.write(f"📊 Probabilitas churn: **{prob:.2f}**")
    st.write("🔍 Hasil Prediksi:", "⚠️ Churn" if pred == 1 else "✅ Tidak Churn")
