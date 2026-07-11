import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")  # Replace with your actual feature names

st.title("Heart stroke prediction by kapil ❤️")
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Up", "Flat", "Down"])

if st.button("predict"):
    raw_input = {
        "Age": age,
        "Sex_"+ sex: 1,
        "ChestPainType" + chest_pain: 1,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG" + resting_ecg: 1,
        "MaxHR": max_hr,
        "ExerciseAngina"+ exercise_angina: 1,
        "Oldpeak": oldpeak,
        "ST_slope" + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the model's expected input
    input_df= input_df[expected_columns]

    # Scale the input data
    input_data_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
     
    if prediction == 1:
        st.error("The model predicts that you are at risk of a heart stroke. Please consult a healthcare professional.")
    else:
        st.success("The model predicts that you are not at risk of a heart stroke.")