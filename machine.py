import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title("🔧 Predictive Maintenance for Industrial Machines")

# -----------------------------
# User Inputs
# -----------------------------
st.header("🧾 Enter Machine Sensor Values")

air_temp = st.number_input("Air temperature [K]", min_value=250.0, max_value=400.0, value=300.0)
process_temp = st.number_input("Process temperature [K]", min_value=250.0, max_value=450.0, value=310.0)
rot_speed = st.number_input("Rotational speed [rpm]", min_value=100.0, max_value=5000.0, value=1500.0)
torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
tool_wear = st.number_input("Tool wear [min]", min_value=0.0, max_value=300.0, value=100.0)

# -----------------------------
# Dummy trained model (for demo)
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Dummy training data
X_dummy = np.random.rand(200, 5)
y_dummy = np.random.randint(0, 2, 200)

scaler = StandardScaler()
X_dummy_scaled = scaler.fit_transform(X_dummy)
model.fit(X_dummy_scaled, y_dummy)

# -----------------------------
# Prediction
# -----------------------------
input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
input_scaled = scaler.transform(input_data)

failure_prob = model.predict_proba(input_scaled)[0][1]
rul = max(0, int(300 - tool_wear))  # simple RUL logic

# Risk category
if failure_prob < 0.3 and rul > 100:
    risk_status = "🟢 SAFE"
elif failure_prob < 0.6 and rul > 50:
    risk_status = "🟠 MEDIUM RISK"
else:
    risk_status = "🔴 HIGH RISK"

# -----------------------------
# Prediction Results
# -----------------------------
st.header("📊 Prediction Results")

st.subheader("Failure Probability")
st.write(f"**{failure_prob:.2f}**")

st.subheader("Remaining Useful Life (RUL)")
st.write(f"**{rul} minutes**")

st.subheader("Overall Machine Status")
st.markdown(f"### {risk_status}")

# -----------------------------
# Feature Importance Plot
# -----------------------------
st.header("📈 Feature Importance (Based on User Input)")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

importances = model.feature_importances_

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(features, importances)

# 🔹 BOLD AXIS LABELS
ax.set_xlabel("User-specific Importance Score", fontsize=12, fontweight="bold")
ax.set_ylabel("Features", fontsize=12, fontweight="bold")

# 🔹 MAKE NUMBERS CLEAR
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=11)

# 🔹 GRID FOR BETTER VISIBILITY
ax.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
st.pyplot(fig)
