import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load trained model and training columns
model = joblib.load("best_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Streamlit UI setup
st.title("🚗 Engine Horsepower Predictor")
st.text("This app predicts estimated engine horsepower based on user input.")

st.title("Now let's put in the car specifications for prediction")

# Input options for categorical variables
cylinder_layout = ['Inline', 'V-type']

# User inputs for top features > 0.05 importance
engine_displacement = st.number_input("Engine Displacement (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
capacity_cm3 = st.number_input("Engine Capacity (cm³)", min_value=500, max_value=8000, value=2000)
turnover_rpm = st.number_input("Max Torque RPM", min_value=500, max_value=10000, value=5000)
cylinder_bore_mm = st.number_input("Cylinder Bore (mm)", min_value=50.0, max_value=150.0, value=85.0)
curb_weight_kg = st.number_input("Curb Weight (kg)", min_value=500, max_value=5000, value=1500)
Year_from = st.slider("Manufacture Year", min_value=1960, max_value=2025, value=2015)
cylinder_layout_input = st.selectbox("Cylinder Layout", cylinder_layout)

# Create input DataFrame
df_input = pd.DataFrame({
    'engine_displacement': [engine_displacement],
    'capacity_cm3': [capacity_cm3],
    'turnover_of_maximum_torque_rpm': [turnover_rpm],
    'cylinder_bore_mm': [cylinder_bore_mm],
    'curb_weight_kg': [curb_weight_kg],
    'Year_from': [Year_from],
    'cylinder_layout_V-type': [1 if cylinder_layout_input == 'V-type' else 0]
})

# Reindex to match model's expected input
input_df = df_input.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Vehicle Horsepower Prediction: {prediction:.2f} HP")

# Background image and style
st.markdown(f''' <style> 
    .stApp {{   
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("https://raw.githubusercontent.com/ka-short/mldp_proj/refs/heads/main/car-wallpaper.jpg");
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
