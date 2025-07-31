# image: https://www.freepik.com/
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load trained model and training columns\
model = joblib.load("best_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Check if the model was loaded correctly
if isinstance(model, list):
    st.error("🚨 The loaded model is a list. You likely did best_rf_model = joblib.dump(...) by mistake when saving.")
    st.stop()

# Streamlit UI setup
st.title("🚗 Engine Horsepower Predictor")
st.text("This app predicts estimated engine horsepower based on user input.")

st.title("Now let's put in the car specifications for prediction")

# Input options for categorical variables
Transmission = ['Manual', 'Automatic', 'CVT']
Drive_Wheels = ['Front', 'Rear', 'All']

# User inputs
curb_weight_kg = st.number_input("Curb Weight (kg)", min_value=500, max_value=5000, value=1500)
capacity_cm3 = st.number_input("Engine Capacity (cm³)", min_value=500, max_value=8000, value=2000)
engine_displacement = st.number_input("Engine Displacement (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
acceleration = st.number_input("0–100 km/h Acceleration (s)", min_value=1.0, max_value=20.0, value=10.0, step=0.1)
max_speed = st.number_input("Max Speed (km/h)", min_value=100, max_value=400, value=220)
transmission = st.selectbox("Select Transmission", Transmission)
drive_wheels = st.selectbox("Select Drive Wheels", Drive_Wheels)

# Create input DataFrame
df_input = pd.DataFrame({
    'curb_weight_kg': [curb_weight_kg],
    'capacity_cm3': [capacity_cm3],
    'engine_displacement': [engine_displacement],
    'acceleration_0_100_km/h_s': [acceleration],
    'max_speed_km_per_h': [max_speed],
    'transmission_Manual': [1 if transmission == 'Manual' else 0],
    'transmission_CVT': [1 if transmission == 'CVT' else 0],
    'drive_wheels_Rear': [1 if drive_wheels == 'Rear' else 0],
    'drive_wheels_All': [1 if drive_wheels == 'All' else 0],
})

# Reindex to match model's expected input
input_df = df_input.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Will this vehicle have high horsepower? Prediction: {prediction:.2f} HP")

# Background image and style
st.markdown(f''' <style> 
    .stApp {{   
        background-image: url();
        background-size: cover;
    }}

    .stButton > button {{
        background-color: #f33;
        color: white;
        border-radius: 15px;
        padding: 10px 20px;
        font-size: 22px;
    }}

    input, select, textarea {{
        background-color: #222 !important;
        color: #fff !important;
    }}

</style>''', unsafe_allow_html=True)
