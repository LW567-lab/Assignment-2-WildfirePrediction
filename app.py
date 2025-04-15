import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("🌲 Wildfire Occurrence Predictor")

st.write("Enter weather conditions and the model will predict whether a wildfire might occur.")

# Input fields for user
temp = st.number_input("🌡 Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
RH = st.number_input("💧 Relative Humidity (%)", min_value=0.0, max_value=100.0, value=40.0)
wind = st.number_input("🍃 Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0)
rain = st.number_input("🌧 Rainfall (mm)", min_value=0.0, max_value=20.0, value=0.0)

# Build feature array (same order as your training data)
X_input = np.array([[temp, RH, wind, rain]])

# Make prediction
if st.button("🔥 Predict Wildfire Occurrence"):
    prediction = model.predict(X_input)
    if prediction[0] == 1:
        st.error("⚠️ A wildfire is likely to occur!")
    else:
        st.success("✅ No wildfire risk detected.")
