import streamlit as st
import joblib
import numpy as np

# Load model and preprocessing tools
model = joblib.load('model/heart_disease_model.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl')

st.title("Heart Disease Risk Predictor")

# Collect user input
user_input = []
for feature in feature_names:
    if feature in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']:
        val = st.number_input(f"{feature}:", min_value=0.0)
    else:
        val = st.selectbox(f"{feature} (0 = No, 1 = Yes):", [0, 1])
    user_input.append(val)

# Make prediction
if st.button("Predict"):
    scaled_input = scaler.transform([user_input])
    prediction = model.predict(scaled_input)[0]
    confidence = model.predict_proba(scaled_input)[0][prediction]
    status = "At Risk" if prediction == 1 else "Not at Risk"
    st.success(f"Prediction: {status} (Confidence: {confidence:.2f})")
