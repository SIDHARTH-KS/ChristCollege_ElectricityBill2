import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained model
model_path = Path(__file__).parent / "electricity_bill_model.pkl"
model = joblib.load(model_path)

# App title
st.title("Electricity Bill Predictor")

st.write(
    "Enter the AC Units and Fan Units "
    "to predict the expected Electric Bill."
)

# User inputs
ac_units = st.number_input(
    "AC Units",
    min_value=0.0,
    step=5.0
)

fan_units = st.number_input(
    "Fan Units",
    min_value=0.0,
    step=5.0
)

# Prediction
if st.button("Predict"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "AC_Units": [ac_units],
        "Fan_Units": [fan_units]
    })

    # Predict using the trained pipeline
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction >= 0:
        st.success(
            f"Expected Electric Bill: ₹{prediction:.2f}"
        )
    else:
        st.error("Error Occurred")
