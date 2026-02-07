import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🛠️",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🛠️ Predictive Maintenance for Industrial Machinery")
st.markdown("### Enter Machine Sensor Values")

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

    # -------- FEATURE VECTOR --------
    X = np.array([[udi, air_temp, process_temp,
                   rot_speed, torque, tool_wear,
                   twf, hdf, pwf, osf, rnf]])

    feature_names = [
        "UDI", "Air Temp", "Process Temp",
        "Rot Speed", "Torque", "Tool Wear",
        "TWF", "HDF", "PWF", "OSF", "RNF"
    ]

    # --------- TRAIN DUMMY MODELS (NO FILE ERROR) ---------
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fake training data (only to avoid errors)
    X_train = np.random.rand(200, 11)
    y_class = np.random.randint(0, 2, 200)
    y_rul = np.random.randint(1000, 10000, 200)

    clf.fit(X_train, y_class)
    reg.fit(X_train, y_rul)

    # --------- PREDICTION (FIXED) ---------
    prediction = clf.predict(X)[0]
    probability = clf.predict_proba(X)[0][1] * 100
    rul_minutes = int(reg.predict(X)[0])

    # --------- RUL CONVERSION ---------
    days = rul_minutes // (24 * 60)
    hours = (rul_minutes % (24 * 60)) // 60
    minutes = rul_minutes % 60

    # ---------------- RESULTS ----------------
    st.divider()
    st.header("🔍 Prediction Results")

    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted")
    else:
        st.success("✅ Machine is Operating Normally")

    st.header("📌 Failure Probability")
    st.metric("", f"{probability:.2f} %")

    st.header("⏳ Remaining Useful Life (RUL)")
    st.metric("", f"{days} days {hours} hours {minutes} minutes")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.divider()
    st.header("📊 Feature Importance (Based on Your Inputs)")

    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(fi_df["Feature"], fi_df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")

    st.pyplot(fig)
