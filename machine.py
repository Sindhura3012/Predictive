import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title("🔧 Predictive Maintenance for Industrial Machinery")
st.subheader("Enter Machine Sensor Values")

# -------------------- LOAD DATA --------------------
df = pd.read_csv("ai4i2020.csv")
df = df.drop(columns=["Product ID", "Type"])

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# -------------------- INPUT SECTION --------------------
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

# -------------------- PREDICTION --------------------
if st.button("🚀 Predict Machine Health"):

    user_input = np.array([[udi, air_temp, process_temp, rot_speed,
                             torque, tool_wear, twf, hdf, pwf, osf, rnf]])

    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted")
    else:
        st.success("✅ Machine is Operating Normally")

    # -------------------- FEATURE IMPORTANCE --------------------
    st.subheader("Feature Importance Based on User Input")

    importance = model.feature_importances_
    features = X.columns

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")

    st.pyplot(fig)
