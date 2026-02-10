import joblib
import pandas as pd
import streamlit as st
import numpy as np
import os

# -------------------------------
# Load Models and Scaler
# -------------------------------

log_reg_model = joblib.load("model/log_reg_model.pkl")
dt_model = joblib.load("model/dt_model.pkl")
knn_model = joblib.load("model/knn_model.pkl")
nb_model = joblib.load("model/nb_model.pkl")
rf_model = joblib.load("model/rf_model.pkl")
xgb_model = joblib.load("model/xgb_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("Heart Disease Prediction App")
st.write("Enter patient details:")

# -------------------------------
# Input Fields
# -------------------------------

age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.slider("Max Heart Rate", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Major Vessels", [0,1,2,3])
thal = st.selectbox("Thalassemia", [0,1,2])

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak,
                             slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    results = {
        "Logistic Regression": log_reg_model.predict(input_scaled)[0],
        "Decision Tree": dt_model.predict(input_scaled)[0],
        "KNN": knn_model.predict(input_scaled)[0],
        "Naive Bayes": nb_model.predict(input_scaled)[0],
        "Random Forest": rf_model.predict(input_scaled)[0],
        "XGBoost": xgb_model.predict(input_scaled)[0]
    }

    st.subheader("Prediction Results")

    for model, pred in results.items():
        label = "Heart Disease" if pred==1 else "No Heart Disease"
        st.write(f"{model}: {label}")

