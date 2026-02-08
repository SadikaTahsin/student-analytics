import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("student_model.pkl")

st.title("🎓 Student Score Predictor")

gender = st.selectbox("Gender", ["Male","Female"])
study = st.slider("Study Hours",1,10,5)
attendance = st.slider("Attendance %",50,100,80)
internet = st.selectbox("Internet Access",["Yes","No"])
previous = st.slider("Previous Score",40,100,70)
age = st.slider("Age",15,20,17)

# Convert to numbers
gender = 0 if gender=="Male" else 1
internet = 1 if internet=="Yes" else 0

input_data = [[gender,age,study,attendance,internet,previous]]

if st.button("Predict Score"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")
