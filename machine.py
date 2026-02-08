import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("🔧 Predictive Maintenance for Industrial Machines")

# ----------------------------------
# User Inputs
# ----------------------------------
st.header("🧾 Enter Sensor Values")

air_temp = st.number_input("Air temperature [K]", 250.0, 400.0, 300.0)
process_temp = st.number_input("Process temperature [K]", 250.0, 450.0, 310.0)
rot_speed = st.number_input("Rotational speed [rpm]", 100.0, 5000.0, 1500.0)
torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
tool_wear = st.number_input("Tool wear [min]", 0.0, 300.0, 100.0)

# ----------------------------------
# Dummy Trained Model
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

failure_prob = model.predict_proba(input_scaled)[0][1]
rul = max(0, int(300 - tool_wear))

if failure_prob < 0.3:
    status = "🟢 SAFE"
elif failure_prob < 0.6:
    status = "🟠 MEDIUM RISK"
else:
    status = "🔴 HIGH RISK"

# ----------------------------------
# Results
# ----------------------------------
st.header("📊 Prediction Results")

col1, col2, col3 = st.columns(3)
col1.metric("Failure Probability", f"{failure_prob:.2f}")
col2.metric("Remaining Useful Life", f"{rul} min")
col3.metric("Machine Status", status)

# ----------------------------------
# Feature Importance Data
# ----------------------------------
st.header("📈 Feature Importance (Horizontal Bar Graph)")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

importance_df = pd.DataFrame({
    "Sensor": features,
    "Importance": model.feature_importances_
})

# ----------------------------------
# Horizontal Bar Chart (Altair)
# ----------------------------------
chart = (
    alt.Chart(importance_df)
    .mark_bar()
    .encode(
        y=alt.Y(
            "Sensor:N",
            sort="-x",
            title="Sensor Values",
            axis=alt.Axis(labelFontSize=13, titleFontSize=16, titleFontWeight="bold")
        ),
        x=alt.X(
            "Importance:Q",
            title="Importance Score",
            axis=alt.Axis(labelFontSize=13, titleFontSize=16, titleFontWeight="bold")
        ),
        tooltip=["Sensor", "Importance"]
    )
    .properties(
        width=900,
        height=350
    )
)

st.altair_chart(chart, use_container_width=True)
