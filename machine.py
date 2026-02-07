import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("ai4i2020.csv")
df = df.drop(columns=["Product ID", "Type"])

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

# App UI
st.title("🔧 Predictive Maintenance for Industrial Machinery")

st.subheader("Enter Machine Sensor Values (with Full Forms)")

col1, col2, col3 = st.columns(3)

with col1:
    udi = st.number_input("UDI – Unique Device Identifier", value=5000.0)
    rot = st.number_input("Rotational Speed (rpm)", value=1500.0)
    twf = st.number_input("TWF – Tool Wear Failure (0 or 1)", value=0.0)
    osf = st.number_input("OSF – Overstrain Failure (0 or 1)", value=0.0)

with col2:
    air = st.number_input("Air Temperature (Kelvin)", value=300.0)
    torque = st.number_input("Torque (Nm)", value=40.0)
    hdf = st.number_input("HDF – Heat Dissipation Failure (0 or 1)", value=0.0)
    rnf = st.number_input("RNF – Random Failure (0 or 1)", value=0.0)

with col3:
    process = st.number_input("Process Temperature (Kelvin)", value=310.0)
    wear = st.number_input("Tool Wear Time (minutes)", value=100.0)
    pwf = st.number_input("PWF – Power Failure (0 or 1)", value=0.0)

if st.button("🚀 Predict Machine Health"):

    user_data = pd.DataFrame([[
        udi, air, process, rot, torque, wear,
        twf, hdf, pwf, osf, rnf
    ]], columns=X.columns)

    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Machine is likely to FAIL")
    else:
        st.success("✅ Machine is HEALTHY")

    # Feature Importance
    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(X.columns, importance)
    ax.set_title("Feature Importance Based on User Input")
    st.pyplot(fig)
