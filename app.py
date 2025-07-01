import streamlit as st
import pandas as pd
import joblib

# Load model, threshold, dan feature names
model = joblib.load("stacking_model.pkl")
threshold = joblib.load("optimal_threshold.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Customer Churn Prediction")
st.write("Masukkan data pelanggan di bawah ini untuk memprediksi kemungkinan churn.")

# Input dari user
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

# Konversi ke DataFrame
input_data = pd.DataFrame([input_dict])


input_data["gender"] = input_data["gender"].map({"Female": 0, "Male": 1})
input_data["Partner"] = input_data["Partner"].map({"Yes": 1, "No": 0})
input_data["Dependents"] = input_data["Dependents"].map({"Yes": 1, "No": 0})
input_data["PhoneService"] = input_data["PhoneService"].map({"Yes": 1, "No": 0})
input_data["InternetService"] = input_data["InternetService"].map({
    "DSL": 0, "Fiber optic": 1, "No": 2
})
input_data["Contract"] = input_data["Contract"].map({
    "Month-to-month": 0, "One year": 1, "Two year": 2
})


input_data = input_data.reindex(columns=feature_names)

# Prediksi
if st.button("Prediksi Churn"):
    prob = model.predict_proba(input_data)[0][1]
    pred = int(prob >= threshold)

    st.write(f"📊 Probabilitas churn: **{prob:.2f}**")
    st.write("🔍 Hasil Prediksi:", "⚠️ Churn" if pred == 1 else "✅ Tidak Churn")
