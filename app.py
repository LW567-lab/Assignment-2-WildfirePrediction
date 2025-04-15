import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Wildfire Predictor", page_icon="🔥")
st.title("🌲 Wildfire Occurrence Predictor")

st.write("Enter weather conditions and the model will predict whether a wildfire might occur.")

# ✅ Debug Step 1: Try loading the model
st.write("🕐 Loading model...")
try:
    model = joblib.load("random_forest_model.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()  # ⛔ Stop execution if model fails

# ✅ Input fields for user
temp = st.number_input("🌡 Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
RH = st.number_input("💧 Relative Humidity (%)", min_value=0.0, max_value=100.0, value=40.0)
wind = st.number_input("🍃 Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0)
rain = st.number_input("🌧 Rainfall (mm)", min_value=0.0, max_value=20.0, value=0.0)

# ✅ Derived features (make sure it matches training order)
temp_squared = temp ** 2
wind_squared = wind ** 2
temp_wind = temp * wind
humidity_wind = RH * wind

X_input = np.array([[temp, RH, wind, rain, temp_squared, wind_squared, temp_wind, humidity_wind]])

# ✅ Prediction
if st.button("🔥 Predict Wildfire Occurrence"):
    try:
        prediction = model.predict(X_input)
        if prediction[0] == 1:
            st.error("⚠️ A wildfire is likely to occur!")
        else:
            st.success("✅ No wildfire risk detected.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
