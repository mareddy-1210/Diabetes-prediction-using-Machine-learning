import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the saved model and scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit app
st.title("Diabetes Prediction App")

# Take user inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=0)
bloodpressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
skinthickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0)
age = st.number_input("Age", min_value=0, max_value=120, value=20)

# Collect input into array
input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness,
                        insulin, bmi, dpf, age]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("The model predicts that the person **has diabetes**.")
    else:
        st.success("The model predicts that the person **does not have diabetes**.")