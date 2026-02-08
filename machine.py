# ============================================
# Predictive Maintenance – Streamlit App
# (Matplotlib-Free Version)
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🛠️",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🛠️ Predictive Maintenance for Industrial Machinery")
st.markdown("### Enter Machine Sensor Values")

# ---------------- LOAD & TRAIN MODELS ----------------
@st.cache_resource
def train_models():
    df = pd.read_csv("ai4i2020.csv")
    df = df.drop(columns=["Product ID", "Type"])

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_scaled, y)

    reg = RandomForestRegressor(n_estimators=150, random_state=42)
    reg.fit(X, df["Tool wear [min]"])

    return clf, reg, scaler, X.columns

clf, reg, scaler, feature_names = train_models()

# ---------------- INPUT UI ----------------
col1, col2, col3 = st.columns(3)

with col1:
    udi = st.number_input("Unique Device Identifier (UDI)", 0.0, 10000.0, 5000.0)
    rot_speed = st.number_input("Rotational Speed (rpm)", 0.0, 5000.0, 1500.0)
    twf = st.number_input("Tool Wear Failure (TWF)", 0.0, 1.0, 0.0)

with col2:
    air_temp = st.number_input("Air Temperature (K)", 250.0, 400.0, 300.0)
    torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0)
    hdf = st.number_input("Heat Dissipation Failure (HDF)", 0.0, 1.0, 0.0)

with col3:
    process_temp = st.number_input("Process Temperature (K)", 250.0, 400.0, 310.0)
    tool_wear = st.number_input("Tool Wear Time (minutes)", 0.0, 300.0, 100.0)
    pwf = st.number_input("Power Failure (PWF)", 0.0, 1.0, 0.0)

osf = st.number_input("Overstrain Failure (OSF)", 0.0, 1.0, 0.0)
rnf = st.number_input("Random Failure (RNF)", 0.0, 1.0, 0.0)

# ---------------- BUTTON ----------------
if st.button("🚀 Predict Machine Health"):

    # -------- INPUT VECTOR --------
    user_input = np.array([[udi, air_temp, process_temp,
                            rot_speed, torque, tool_wear,
                            twf, hdf, pwf, osf, rnf]])

    user_scaled = scaler.transform(user_input)

    # -------- PREDICTIONS --------
    failure_prob = clf.predict_proba(user_scaled)[0][1] * 100
    rul_minutes = int(reg.predict(user_input)[0])

    # -------- RUL FORMAT --------
    days = rul_minutes // (24 * 60)
    hours = (rul_minutes % (24 * 60)) // 60
    minutes = rul_minutes % 60

    # -------- RISK LOGIC --------
    if failure_prob >= 70 or rul_minutes <= 2000:
        risk = "HIGH RISK"
        color = "🔴"
    elif failure_prob >= 30 or rul_minutes <= 5000:
        risk = "MEDIUM RISK"
        color = "🟠"
    else:
        risk = "SAFE"
        color = "🟢"

    # ---------------- RESULTS ----------------
    st.divider()
    st.header("🔍 Prediction Results")

    st.subheader(f"{color} Machine Status: **{risk}**")

    st.header("📌 Failure Probability")
    st.metric("", f"{failure_prob:.2f} %")

    st.header("⏳ Remaining Useful Life (RUL)")
    st.metric("", f"{days} days {hours} hours {minutes} minutes")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.divider()
    st.header("📊 Feature Importance (Based on Your Inputs)")

    global_importance = clf.feature_importances_
    normalized_input = np.abs(user_input[0]) / (np.sum(np.abs(user_input[0])) + 1e-6)
    user_importance = global_importance * normalized_input

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": user_importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))
