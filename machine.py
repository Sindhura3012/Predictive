import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title("🔧 Predictive Maintenance for Industrial Machinery")
st.subheader("Enter Machine Sensor Values")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("ai4i2020.csv")
df = df.drop(columns=["Product ID", "Type"])

# Classification (Failure)
X_cls = df.drop(columns=["Machine failure"])
y_cls = df["Machine failure"]

# Regression (RUL approximation using tool wear)
X_rul = X_cls
y_rul = 10000 - df["Tool wear [min]"]  # Synthetic RUL target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cls)

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_scaled, y_cls)

rul_model = RandomForestRegressor(n_estimators=150, random_state=42)
rul_model.fit(X_scaled, y_rul)

# ---------------- INPUT UI ----------------
col1, col2, col3 = st.columns(3)

with col1:
    udi = st.number_input("Unique Device Identifier (UDI)", value=5000.0)
    rot_speed = st.number_input("Rotational Speed (rpm)", value=1500.0)
    twf = st.number_input("Tool Wear Failure (TWF)", min_value=0.0, max_value=1.0, value=0.0)

with col2:
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    torque = st.number_input("Torque (Nm)", value=40.0)
    hdf = st.number_input("Heat Dissipation Failure (HDF)", min_value=0.0, max_value=1.0, value=0.0)

with col3:
    process_temp = st.number_input("Process Temperature (K)", value=310.0)
    tool_wear = st.number_input("Tool Wear Time (min)", value=100.0)
    pwf = st.number_input("Power Failure (PWF)", min_value=0.0, max_value=1.0, value=0.0)

osf = st.number_input("Overstrain Failure (OSF)", min_value=0.0, max_value=1.0, value=0.0)
rnf = st.number_input("Random Failure (RNF)", min_value=0.0, max_value=1.0, value=0.0)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Machine Health"):

    user_input = np.array([[udi, air_temp, process_temp, rot_speed,
                             torque, tool_wear, twf, hdf, pwf, osf, rnf]])

    user_scaled = scaler.transform(user_input)

    # Failure prediction
    prediction = clf.predict(user_scaled)[0]
    probability = clf.predict_proba(user_scaled)[0][1] * 100

    # RUL prediction (minutes)
    rul_minutes = int(rul_model.predict(user_scaled)[0])
    rul_minutes = max(rul_minutes, 0)

    # Convert RUL
    days = rul_minutes // 1440
    hours = (rul_minutes % 1440) // 60
    minutes = rul_minutes % 60

    # ---------------- RESULTS ----------------
    st.subheader("🔍 Prediction Results")

    if prediction == 1:
        st.error(f"⚠️ Machine Failure Predicted")
    else:
        st.success("✅ Machine is Operating Normally")

    st.metric("Failure Probability", f"{probability:.2f} %")
    st.metric("Remaining Useful Life (RUL)",
              f"{days} days {hours} hours {minutes} minutes")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("📊 Feature Importance")

    importance = clf.feature_importances_
    features = X_cls.columns

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance Based on User Input")

    st.pyplot(fig)
