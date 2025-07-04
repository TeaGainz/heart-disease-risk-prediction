import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load('model/heart_disease_model.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl') 

st.title("Heart Disease Risk Predictor")

# Define mappings
age_map = {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5,
           '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80 or older': 12}
gen_health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
diabetic_map = {'No': 0, 'No, borderline diabetes': 1, 'Yes (during pregnancy)': 2, 'Yes': 3}

# --- Collect Inputs ---
st.subheader("Basic Info")
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.selectbox("Age Category", list(age_map.keys()))
race = st.selectbox("Race", ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 
                             'Hispanic', 'Other']) 

st.subheader("Health Info")
bmi = st.number_input("BMI", 10.0, 100.0, step=0.1)
physical = st.slider("Physical Health (last 30 days)", 0, 30, 0)
mental = st.slider("Mental Health (last 30 days)", 0, 30, 0)
sleep = st.slider("Sleep Time (hrs/day)", 0, 24, 7)

st.subheader("Binary Questions (Yes = 1, No = 0)")
binary_features = [
    "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "PhysicalActivity",
    "Asthma", "KidneyDisease", "SkinCancer",
]
binary_input = {}
for feat in binary_features:
    binary_input[feat] = st.selectbox(f"{feat}?", [0, 1])

st.subheader("Other Ordinal Features")
gen_health = st.selectbox("General Health", list(gen_health_map.keys()))
diabetic = st.selectbox("Diabetic", list(diabetic_map.keys()))

# --- Build DataFrame ---
data = {feat: 0 for feat in feature_names if feat != "HeartDisease"}

# Fill numeric & ordinal
data['BMI'] = bmi
data['PhysicalHealth'] = physical
data['MentalHealth'] = mental
data['SleepTime'] = sleep
data['Sex'] = 1 if sex == 'Female' else 0
data['AgeCategory'] = age_map[age]
data['GenHealth'] = gen_health_map[gen_health]
data['Diabetic'] = diabetic_map[diabetic]

# Fill binary
for feat in binary_features:
    data[feat] = binary_input[feat]

# Handle one-hot encoded race column
race_column = f"Race_{race}"
if race_column in data:
    data[race_column] = 1

# --- final DataFrame ---
input_df = pd.DataFrame([data])
# Scale numeric columns
numeric_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("### Prediction:")
    st.success("At Risk" if prediction == 1 else "Not At Risk")
    st.markdown(f"### Confidence: `{prob:.2%}`")
