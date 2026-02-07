import streamlit as st
import numpy as np
import joblib

# Load trained models
clf = joblib.load("failure_model.pkl")
reg = joblib.load("rul_model.pkl")
scaler = joblib.load("scaler.pkl")

# Utility functions
def prob_to_percentage(prob):
    return round(prob * 100, 2)

def format_rul(minutes):
    minutes = int(minutes)
    days = minutes // (24 * 60)
    hours = (minutes % (24 * 60)) // 60
    mins = minutes % 60
    return f"{days} days {hours} hours {mins} minutes"

st.title("🛠 Predictive Maintenance System")

st.subheader("Enter Machine Sensor Values")

# ---- SENSOR FULL FORMS ----
UDI = st.number_input("UDI – Unique Device Identifier", value=5000.0)
air_temp = st.number_input("Air Temperature [K]", value=300.0)
process_temp = st.number_input("Process Temperature [K]", value=310.0)
rpm = st.number_input("Rotational Speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [minutes]", value=100.0)

TWF = st.number_input("TWF – Tool Wear Failure (0 or 1)", value=0.0)
HDF = st.number_input("HDF – Heat Dissipation Failure (0 or 1)", value=0.0)
PWF = st.number_input("PWF – Power Failure (0 or 1)", value=0.0)
OSF = st.number_input("OSF – Overstrain Failure (0 or 1)", value=0.0)
RNF = st.number_input("RNF – Random Failure (0 or 1)", value=0.0)

if st.button("🚀 Predict Machine Health"):

    input_data = np.array([[UDI, air_temp, process_temp, rpm, torque,
                             tool_wear, TWF, HDF, PWF, OSF, RNF]])

    scaled_input = scaler.transform(input_data)

    # Predictions
    failure_prob = clf.predict_proba(scaled_input)[0][1]
    rul_minutes = reg.predict(input_data)[0]

    st.success("Prediction Successful")

    st.metric(
        label="Failure Probability",
        value=f"{prob_to_percentage(failure_prob)} %"
    )

    st.metric(
        label="Remaining Useful Life (RUL)",
        value=format_rul(rul_minutes)
    )
