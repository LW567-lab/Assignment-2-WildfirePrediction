import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Wildfire Predictor", page_icon="🔥")
st.title("🌲 Wildfire Occurrence Predictor")

st.write("Enter temperature, humidity, and wind conditions to predict wildfire risk.")

# 输入原始特征
temp = st.slider("🌡 Temperature (°C)", 0.0, 50.0, 20.0)
RH = st.slider("💧 Relative Humidity (%)", 0.0, 100.0, 40.0)
wind = st.slider("🍃 Wind Speed (km/h)", 0.0, 50.0, 10.0)

# 派生特征计算
temp_squared = temp ** 2
wind_squared = wind ** 2
temp_wind = temp * wind
humidity_wind = RH * wind

# 构建 DataFrame，确保列名顺序完全一致
X_input = pd.DataFrame([[
    temp, RH, wind,
    temp_squared, wind_squared, temp_wind, humidity_wind
]], columns=[
    'temp', 'RH', 'wind',
    'temp_squared', 'wind_squared', 'temp_wind', 'humidity_wind'
])

# 加载模型并预测
try:
    model = joblib.load("random_forest_model.pkl")
    st.success("✅ Model loaded successfully!")

    if st.button("🔥 Predict Wildfire"):
        prediction = model.predict(X_input)
        if prediction[0] == 1:
            st.error("⚠️ A wildfire is likely to occur!")
        else:
            st.success("✅ No wildfire risk detected.")
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
