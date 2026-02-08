import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
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

# ---------------- RUL FORMAT FUNCTION ----------------
def format_rul(minutes):
    minutes = int(max(minutes, 0))
    days = minutes // (24 * 60)
    hours = (minutes % (24 * 60)) // 60
    mins = minutes % 60
    return f"{days} days {hours} hours {mins} minutes"

# ---------------- LOAD & TRAIN MODEL ----------------
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

    return clf, reg, scaler, X.columns.tolist()

clf, reg, scaler, feature_names = train_models()

# ---------------- INPUT UI ----------------
c1, c2, c3 = st.columns(3)

with c1:
    udi = st.number_input("Unique Device Identifier (UDI)", value=5000.0)
    rot_speed = st.number_input("Rotational Speed (rpm)", value=1500.0)
    twf = st.number_input("Tool Wear Failure (TWF)", value=0.0)

with c2:
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    torque = st.number_input("Torque (Nm)", value=40.0)
    hdf = st.number_input("Heat Dissipation Failure (HDF)", value=0.0)

with c3:
    process_temp = st.number_input("Process Temperature (K)", value=310.0)
    tool_wear = st.number_input("Tool Wear Time (minutes)", value=100.0)
    pwf = st.number_input("Power Failure (PWF)", value=0.0)

osf = st.number_input("Overstrain Failure (OSF)", value=0.0)
rnf = st.number_input("Random Failure (RNF)", value=0.0)

# ---------------- BUTTON ----------------
if st.button("🚀 Predict Machine Health"):

    user_input = np.array([[udi, air_temp, process_temp,
                            rot_speed, torque, tool_wear,
                            twf, hdf, pwf, osf, rnf]])

    user_scaled = scaler.transform(user_input)

    # ---------------- PREDICTIONS ----------------
    failure_prob = clf.predict_proba(user_scaled)[0][1] * 100
    rul_minutes = reg.predict(user_input)[0]

    # ---------------- RISK LOGIC ----------------
    if failure_prob < 30 and rul_minutes > 5000:
        risk = "SAFE"
        color = "🟢"
    elif failure_prob < 70 and rul_minutes > 2000:
        risk = "MEDIUM RISK"
        color = "🟠"
    else:
        risk = "HIGH RISK"
        color = "🔴"

    # ---------------- RESULTS ----------------
    st.divider()
    st.header("🔍 Prediction Results")

    st.subheader(f"{color} Machine Status: **{risk}**")

    st.markdown("## 📌 Failure Probability")
    st.metric("", f"{failure_prob:.2f} %")

    st.markdown("## ⏳ Remaining Useful Life (RUL)")
    st.metric("", format_rul(rul_minutes))

    # ---------------- FEATURE IMPORTANCE ----------------
    st.divider()
    st.header("📊 Feature Importance (Based on User Input)")

    global_importance = clf.feature_importances_
    normalized_input = np.abs(user_input[0]) / (np.sum(np.abs(user_input[0])) + 1e-6)
    user_importance = global_importance * normalized_input

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": user_importance
    }).sort_values("Importance", ascending=False)

    chart = alt.Chart(fi_df).mark_bar().encode(
        x=alt.X("Importance:Q", title="User-specific Importance Score"),
        y=alt.Y("Feature:N", sort="-x", title=""),
        tooltip=["Feature", "Importance"]
    ).properties(height=420)

    st.altair_chart(chart, use_container_width=True)
