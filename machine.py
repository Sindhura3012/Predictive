import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Predictive Maintenance System",
    layout="wide"
)

st.title("🔧 Predictive Maintenance for Industrial Machinery")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ai4i2020.csv")

df = load_data()

# -------------------------------
# TARGET DETECTION
# -------------------------------
failure_candidates = ["Machine failure", "Failure", "Target", "Label"]
rul_candidates = ["RUL", "Remaining Useful Life"]

failure_col = next((c for c in failure_candidates if c in df.columns), None)
rul_col = next((c for c in rul_candidates if c in df.columns), None)

# -------------------------------
# FEATURE SELECTION
# -------------------------------
numeric_df = df.select_dtypes(include=[np.number])

drop_cols = []
if failure_col:
    drop_cols.append(failure_col)
if rul_col:
    drop_cols.append(rul_col)

X = numeric_df.drop(columns=drop_cols, errors="ignore")
X = X.fillna(X.median())

# -------------------------------
# ADD OPERATING TIME
# -------------------------------
if "Operating_Time" not in X.columns:
    X["Operating_Time"] = np.arange(len(X))
    df["Operating_Time"] = X["Operating_Time"]

# -------------------------------
# FAILURE MODEL
# -------------------------------
rf_clf = None
if failure_col:
    y_fail = df[failure_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_fail, test_size=0.2, random_state=42, stratify=y_fail
    )

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    rf_clf.fit(X_train, y_train)

# -------------------------------
# RUL GENERATION
# -------------------------------
if rul_col is None:
    df["RUL"] = df["Operating_Time"].max() - df["Operating_Time"]
    rul_col = "RUL"

# -------------------------------
# RUL MODEL
# -------------------------------
y_rul = df[rul_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_rul, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
rf_reg.fit(X_train, y_train)

# -------------------------------
# INPUT SAFETY
# -------------------------------
def clip_input(val, col):
    min_v = df[col].quantile(0.01)
    max_v = df[col].quantile(0.99)
    exceeded = val < min_v or val > max_v
    return np.clip(val, min_v, max_v), exceeded

def build_input(user_inputs):
    row = {}
    warnings = []

    for col in X.columns:
        if col in user_inputs:
            val, exceeded = clip_input(user_inputs[col], col)
            row[col] = val
            if exceeded:
                warnings.append(col)
        else:
            row[col] = df[col].median()

    return pd.DataFrame([row]), warnings

# -------------------------------
# USER INPUT
# -------------------------------
st.sidebar.header("Machine Input Parameters")

user_inputs = {}
for col in X.columns:
    user_inputs[col] = st.sidebar.number_input(
        col,
        value=float(df[col].median())
    )

# -------------------------------
# PREDICTION
# -------------------------------
if st.sidebar.button("Predict Machine Health"):

    input_df, warn_cols = build_input(user_inputs)

    failure_prob = None
    if rf_clf:
        failure_prob = rf_clf.predict_proba(input_df)[0][1]

    rul_pred = rf_reg.predict(input_df)[0]
    rul_minutes = max(rul_pred, 0)

    if failure_prob is not None:
        if failure_prob >= 0.7 or rul_minutes < 30:
            status = "🔴 HIGH RISK"
        elif failure_prob >= 0.4 or rul_minutes < 100:
            status = "🟠 MEDIUM RISK"
        else:
            status = "🟢 SAFE"
    else:
        status = "🟢 SAFE"

    st.subheader("Prediction Results")

    if failure_prob is not None:
        st.metric("Failure Probability", f"{failure_prob:.2f}")

    st.metric("Predicted RUL", f"{rul_minutes:.2f} minutes")
    st.markdown(f"### Machine Status: {status}")

    if warn_cols:
        st.warning(
            f"Input exceeds training range for: {', '.join(warn_cols)}. "
            "Values were safely adjusted."
        )

    st.subheader("Feature Importance (RUL Model)")

    fig, ax = plt.subplots()
    imp = rf_reg.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": imp
    }).sort_values(by="Importance")

    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    st.pyplot(fig)
