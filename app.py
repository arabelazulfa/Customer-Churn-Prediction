import streamlit as st
import pandas as pd
import joblib

# Load model, threshold, dan feature names
model = joblib.load("stacking_model.pkl")
threshold = joblib.load("optimal_threshold.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Customer Churn Prediction")
st.write("Masukkan data pelanggan di bawah ini untuk memprediksi kemungkinan churn.")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 3000.0)

# Buat input dict
input_dict = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "InternetService": internet_service,
    "Contract": contract,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

# Konversi jadi DataFrame
input_data = pd.DataFrame([input_dict])

# Pastikan kolom urut dan lengkap
input_data = input_data.reindex(columns=feature_names)

if st.button("Prediksi Churn"):
    prob = model.predict_proba(input_data)[0][1]
    pred = int(prob >= threshold)

    st.write(f"📊 Probabilitas churn: **{prob:.2f}**")
    st.write("🔍 Hasil Prediksi:", "⚠️ Churn" if pred == 1 else "✅ Tidak Churn")
