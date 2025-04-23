import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math

st.set_page_config(page_title="Wildfire Predictor", page_icon="🔥")
st.title("🌲 Wildfire Occurrence Predictor")

st.write("Enter environmental conditions to predict the risk of wildfire occurrence.")

# Input fields — must match training features
ffmc = st.slider("FFMC (Fine Fuel Moisture Code)", 0.0, 100.0, 85.0)
dmc = st.slider("DMC (Duff Moisture Code)", 0.0, 300.0, 100.0)
dc = st.slider("DC (Drought Code)", 0.0, 800.0, 300.0)
isi = st.slider("ISI (Initial Spread Index)", 0.0, 50.0, 10.0)
temp = st.slider("🌡 Temperature (°C)", 0.0, 50.0, 20.0)
RH = st.slider("💧 Relative Humidity (%)", 0.0, 100.0, 45.0)
wind = st.slider("🍃 Wind Speed (km/h)", 0.0, 100.0, 10.0)
month = st.selectbox("📅 Month", list(range(1, 13)))
day = st.selectbox("📆 Day of Week (1=Mon, 7=Sun)", list(range(1, 8)))

# Encode cyclical features
month_sin = math.sin(2 * math.pi * month / 12)
month_cos = math.cos(2 * math.pi * month / 12)
day_sin = math.sin(2 * math.pi * day / 7)
day_cos = math.cos(2 * math.pi * day / 7)

X_input = pd.DataFrame([[
    ffmc, dmc, dc, isi, temp, RH, wind,
    month, day, month_sin, month_cos, day_sin, day_cos
]], columns=[
    'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind',
    'month', 'day', 'month_sin', 'month_cos', 'day_sin', 'day_cos'
])

# Load model and predict
model = joblib.load("random_forest_model.pkl")
if st.button("🔥 Predict"):
    prediction = model.predict(X_input)
    if prediction[0] == 1:
        st.error("⚠️ A wildfire is likely to occur!")
    else:
        st.success("✅ No wildfire risk detected.")
