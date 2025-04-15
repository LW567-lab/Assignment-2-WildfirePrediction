import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("random_forest_model.pkl")

st.title("🌲 Wildfire Occurrence Predictor")

st.write("Enter weather conditions and the model will predict whether a wildfire might occur.")

# Base inputs
temp = st.number_input("🌡 Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
RH = st.number_input("💧 Relative Humidity (%)", min_value=0.0, max_value=100.0, value=40.0)
wind = st.number_input("🍃 Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0)
rain = st.number_input("🌧 Rainfall (mm)", min_value=0.0, max_value=20.0, value=0.0)

# Derived features
temp_squared = temp ** 2
wind_squared = wind ** 2
temp_wind = temp * wind
humidity_wind = RH * wind

# Feature array (same order as training)
X_input = np.array([[temp, RH, wind, rain, temp_squared, wind_squared, temp_wind, humidity_wind]])

# Predict
if st.button("🔥 Predict Wildfire Occurrence"):
    prediction = model.predict(X_input)
    if prediction[0] == 1:
        st.error("⚠️ A wildfire is likely to occur!")
    else:
        st.success("✅ No wildfire risk detected.")
