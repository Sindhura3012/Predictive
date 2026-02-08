import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("🔧 Predictive Maintenance for Industrial Machines")

# ----------------------------------
# User Inputs
# ----------------------------------
st.header("🧾 Enter Machine Sensor Values")

air_temp = st.number_input("Air temperature [K]", 250.0, 400.0, 300.0)
process_temp = st.number_input("Process temperature [K]", 250.0, 450.0, 310.0)
rot_speed = st.number_input("Rotational speed [rpm]", 100.0, 5000.0, 1500.0)
torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
tool_wear = st.number_input("Tool wear [min]", 0.0, 300.0, 100.0)

# ----------------------------------
# Dummy trained model
# ----------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)

X_dummy = np.random.rand(300, 5)
y_dummy = np.random.randint(0, 2, 300)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummy)
model.fit(X_scaled, y_dummy)

# ----------------------------------
# Prediction
# ----------------------------------
input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
input_scaled = scaler.transform(input_data)

failure_probability = model.predict_proba(input_scaled)[0][1]
rul = max(0, int(300 - tool_wear))

# Risk logic
if failure_probability < 0.3 and rul > 100:
    risk = "🟢 SAFE"
elif failure_probability < 0.6 and rul > 50:
    risk = "🟠 MEDIUM RISK"
else:
    risk = "🔴 HIGH RISK"

# ----------------------------------
# Results
# ----------------------------------
st.header("📊 Prediction Results")

st.subheader("Failure Probability")
st.metric(label="", value=f"{failure_probability:.2f}")

st.subheader("Remaining Useful Life (RUL)")
st.metric(label="", value=f"{rul} minutes")

st.subheader("Overall Machine Status")
st.markdown(f"## {risk}")

# ----------------------------------
# Feature Importance (NO matplotlib)
# ----------------------------------
st.header("📈 Feature Importance (Based on User Input)")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

importance_df = pd.DataFrame({
    "Features": features,
    "Importance Score": model.feature_importances_
}).set_index("Features")

st.bar_chart(importance_df)
