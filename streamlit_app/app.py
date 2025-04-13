import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("🌲 Wildfire Occurrence Predictor")
st.write("Enter weather conditions and the model will predict whether a wildfire might occur.")

# Input fields
temp = st.number_input("🌡 Temperature (°C)", value=20.0)
RH = st.number_input("💧 Relative Humidity (%)", value=40.0)
wind = st.number_input("🍃 Wind Speed (km/h)", value=10.0)
rain = st.number_input("🌧 Rainfall (mm)", value=0.0)

# Derived features
temp_squared = temp ** 2
wind_squared = wind ** 2
temp_wind = temp * wind
humidity_wind = RH * wind

# Build input vector (must match model training feature order)
X_input = np.array([[temp, RH, wind, rain, temp_squared, wind_squared, temp_wind, humidity_wind]])

# Prediction
if st.button("🔥 Predict Wildfire Occurrence"):
    prediction = model.predict(X_input)
    if prediction[0] == 1:
        st.error("⚠️ A wildfire is likely to occur!")
    else:
        st.success("✅ No wildfire risk detected.")
