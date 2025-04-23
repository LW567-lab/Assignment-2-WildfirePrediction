import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Wildfire Predictor", page_icon="🔥")
st.title("🔥 Wildfire Occurrence Predictor")

st.markdown("Enter **temperature**, **humidity**, and **wind** conditions to predict wildfire risk.")

temp = st.slider("🌡 Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
RH = st.slider("💧 Relative Humidity (%)", min_value=0.0, max_value=100.0, value=40.0)
wind = st.slider("🍃 Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0)

temp_squared = temp ** 2
wind_squared = wind ** 2
temp_wind = temp * wind
humidity_wind = RH * wind

X_input = np.array([[temp, RH, wind, temp_squared, wind_squared, temp_wind, humidity_wind]])

try:
    model = joblib.load("random_forest_model.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

if st.button("🔥 Predict Wildfire"):
    prediction = model.predict(X_input)
    probability = model.predict_proba(X_input)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ A wildfire is likely to occur! (Risk: {probability:.2%})")
    else:
        st.success(f"✅ No wildfire risk detected. (Risk: {probability:.2%})")
