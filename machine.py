import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("🔧 Predictive Maintenance for Industrial Machinery")

# -------------------------------
# RUL formatter
# -------------------------------
def format_rul(minutes):
    minutes = int(max(minutes, 0))
    days = minutes // (24 * 60)
    hours = (minutes % (24 * 60)) // 60
    mins = minutes % 60
    return f"{days} days {hours} hours {mins} minutes"

# -------------------------------
# Load dataset & train
# -------------------------------
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

# -------------------------------
# USER INPUT UI
# -------------------------------
st.subheader("Enter Machine Sensor Values")

cols = st.columns(3)
user_values = []

for i, col in enumerate(X.columns):
    with cols[i % 3]:
        val = st.number_input(col, value=float(X[col].mean()))
        user_values.append(val)

user_input = np.array(user_values).reshape(1, -1)
user_scaled = scaler.transform(user_input)

# -------------------------------
# Predict button
# -------------------------------
if st.button("🚀 Predict Machine Health"):
    failure_prob = clf.predict_proba(user_scaled)[0][1] * 100
    rul_minutes = reg.predict(user_input)[0]

    # USER-SPECIFIC FEATURE IMPORTANCE
    global_importance = clf.feature_importances_
    normalized_input = np.abs(user_input[0]) / (np.sum(np.abs(user_input[0])) + 1e-6)
    user_importance = global_importance * normalized_input

    st.success("Prediction Completed")

    c1, c2 = st.columns(2)
    c1.metric("Failure Probability", f"{failure_prob:.2f}%")
    c2.metric("Remaining Useful Life", format_rul(rul_minutes))

    if failure_prob < 30:
        st.success("Machine Status: SAFE")
    elif failure_prob < 70:
        st.warning("Machine Status: WARNING")
    else:
        st.error("Machine Status: HIGH RISK")

    # -------------------------------
    # Feature Importance Plot
    # -------------------------------
    st.subheader("Feature Importance Based on User Input")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(X.columns, user_importance)
    ax.set_xlabel("User-specific Importance Score")
    st.pyplot(fig)
