import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("appliance_energy_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Appliance Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Title
st.title("⚡ Appliance Energy Consumption Predictor")

st.write(
    "Predict household appliance energy consumption using a Machine Learning model trained on environmental sensor and weather data."
)

st.subheader("📌 Project Overview")

st.write("""
This application predicts household appliance energy consumption using a
Random Forest Regressor trained on environmental sensor, weather, and
time-based features.

The goal is to support smarter energy management and improve household
energy efficiency.
""")

# Default feature values (dataset means)
DEFAULTS = {
    "lights": 3.801875,
    "T1": 21.686571,
    "RH_1": 40.259739,
    "T2": 20.341219,
    "RH_2": 40.420420,
    "T3": 22.267611,
    "RH_3": 39.242500,
    "T4": 20.855335,
    "RH_4": 39.026904,
    "T5": 19.592106,
    "RH_5": 50.949283,
    "T6": 7.910939,
    "RH_6": 54.609083,
    "T7": 20.267106,
    "RH_7": 35.388200,
    "T8": 22.029107,
    "RH_8": 42.936165,
    "T9": 19.485828,
    "RH_9": 41.552401,
    "T_out": 7.412580,
    "Press_mm_hg": 755.522602,
    "RH_out": 79.750418,
    "Windspeed": 4.039752,
    "Visibility": 38.330834,
    "Tdewpoint": 3.760995,
    "rv1": 24.988033,
    "rv2": 24.988033,
    "hour": 11.502002,
    "day": 16.057411,
    "month": 3.101647,
    "weekday": 2.977249,
    "is_weekend": 0.277274,
    "avg_temp": 19.381758,
    "avg_humidity": 42.709411
}

# Sidebar
st.sidebar.header("Input Features")

lights = st.sidebar.slider(
    "Lights Usage",
    min_value=0,
    max_value=70,
    value=5
)

hour = st.sidebar.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=12
)

month = st.sidebar.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=3
)

avg_temp = st.sidebar.slider(
    "Average Temperature (°C)",
    min_value=0.0,
    max_value=40.0,
    value=19.4,
    step=0.1
)

avg_humidity = st.sidebar.slider(
    "Average Humidity (%)",
    min_value=20.0,
    max_value=100.0,
    value=42.7,
    step=0.1
)

rh_out = st.sidebar.slider(
    "Outdoor Humidity (%)",
    min_value=20.0,
    max_value=100.0,
    value=79.8,
    step=0.1
)

windspeed = st.sidebar.slider(
    "Wind Speed",
    min_value=0.0,
    max_value=20.0,
    value=4.0,
    step=0.1
)

# Create input data
input_data = DEFAULTS.copy()

input_data["lights"] = lights
input_data["hour"] = hour
input_data["month"] = month
input_data["avg_temp"] = avg_temp
input_data["avg_humidity"] = avg_humidity
input_data["RH_out"] = rh_out
input_data["Windspeed"] = windspeed

input_df = pd.DataFrame([input_data])

# Prediction
if st.button("🔮 Predict Energy Consumption"):

    prediction = model.predict(input_df)[0]

    st.metric(
        label="Predicted Appliance Energy Consumption",
        value=f"{prediction:.2f} Wh"
    )

    if prediction < 100:
        st.success("🟢 Low Energy Consumption")

    elif prediction < 250:
        st.warning("🟡 Moderate Energy Consumption")

    else:
        st.error("🔴 High Energy Consumption")

    st.subheader("Recommendations")

    if prediction > 250:
        st.write("💡 Reduce unnecessary lighting usage")
        st.write("🌡️ Optimize temperature settings")
        st.write("⏰ Shift appliance usage to off-peak hours")
        st.write("📊 Monitor humidity conditions")

    elif prediction > 100:
        st.write("✅ Maintain current energy usage patterns")
        st.write("💡 Consider reducing lighting when not required")

    else:
        st.write("🎉 Excellent energy efficiency")

st.markdown("---")

st.subheader("Model Information")

st.write("""
**Model:** Random Forest Regressor

**Features:** 34

**Target:** Appliance Energy Consumption (Wh)

**Dataset:** ~19,700 observations

**Goal:** Predict household appliance energy usage for smarter energy management.
""")