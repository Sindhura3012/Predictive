import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Predictive Maintenance", layout="centered")

st.title("🔧 Predictive Maintenance for Industrial Machinery")
st.write("Machine Failure Prediction using Machine Learning")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    return df

df = load_data()

# -----------------------------
# Feature selection
# -----------------------------
features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

target = 'Machine failure'

X = df[features]
y = df[target]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# Model evaluation
# -----------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {accuracy:.4f}")

# -----------------------------
# User Input
# -----------------------------
st.subheader("🧮 Enter Machine Sensor Values")

air_temp = st.number_input("Air Temperature (K)", min_value=200.0, max_value=400.0, value=300.0)
process_temp = st.number_input("Process Temperature (K)", min_value=200.0, max_value=450.0, value=310.0)
rpm = st.number_input("Rotational Speed (rpm)", min_value=0.0, max_value=5000.0, value=1500.0)
torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=200.0, value=40.0)
tool_wear = st.number_input("Tool Wear (min)", min_value=0.0, max_value=300.0, value=100.0)

input_data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])
input_scaled = scaler.transform(input_data)

# -----------------------------
# Range warning
# -----------------------------
for i, col in enumerate(features):
    min_val = X[col].min()
    max_val = X[col].max()
    if input_data[0][i] < min_val or input_data[0][i] > max_val:
        st.warning(f"⚠️ {col} is outside training range")

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Machine Failure"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"❌ Machine Failure Predicted")
    else:
        st.success(f"✅ No Failure Predicted")

    st.write(f"**Failure Probability:** {probability:.2%}")

# -----------------------------
# Feature importance
# -----------------------------
st.subheader("📌 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Predictive Maintenance | Machine Learning | Streamlit App")
