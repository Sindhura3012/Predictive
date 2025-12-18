import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title("🔧 Predictive Maintenance for Industrial Machinery")
st.caption("Failure Prediction • RUL Estimation • Risk Assessment")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ai4i2020.csv")

df = load_data()

# --------------------------------------------------
# TARGET DETECTION
# --------------------------------------------------
failure_col = "Machine failure" if "Machine failure" in df.columns else None
rul_col = "RUL" if "RUL" in df.columns else None

numeric_df = df.select_dtypes(include=np.number)

X = numeric_df.drop(columns=[c for c in [failure_col, rul_col] if c], errors="ignore")
X = X.fillna(X.median())

# --------------------------------------------------
# CREATE RUL IF NOT PRESENT
# --------------------------------------------------
if rul_col is None:
    df["Operating_Time"] = np.arange(len(df))
    df["RUL"] = df["Operating_Time"].max() - df["Operating_Time"]
    rul_col = "RUL"

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------
if failure_col:
    y_fail = df[failure_col]
    Xtr, Xte, ytr, yte = train_test_split(X, y_fail, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
else:
    clf = None

y_rul = df[rul_col]
Xtr, Xte, ytr, yte = train_test_split(X, y_rul, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(Xtr, ytr)

# --------------------------------------------------
# INPUT RANGE CHECK
# --------------------------------------------------
def validate_inputs(val, col):
    lo, hi = df[col].min(), df[col].max()
    return lo <= val <= hi

# --------------------------------------------------
# SIDEBAR FORM (NO BLUR)
# --------------------------------------------------
st.sidebar.header("⚙ Machine Parameters")

with st.sidebar.form("prediction_form"):
    user_inputs = {}
    warnings = []

    for col in X.columns:
        val = st.number_input(col, value=float(df[col].median()))
        user_inputs[col] = val
        if not validate_inputs(val, col):
            warnings.append(col)

    submit = st.form_submit_button("🚀 Predict Machine Health")

# --------------------------------------------------
# MAIN OUTPUT
# --------------------------------------------------
tab1, tab2 = st.tabs(["📊 Prediction", "📈 Feature Importance"])

with tab1:
    if submit:
        input_df = pd.DataFrame([user_inputs])

        failure_prob = clf.predict_proba(input_df)[0][1] if clf else 0
        rul_pred = max(reg.predict(input_df)[0], 0)

        if failure_prob > 0.7 or rul_pred < 30:
            status, color = "HIGH RISK", "🔴"
        elif failure_prob > 0.4 or rul_pred < 100:
            status, color = "MEDIUM RISK", "🟠"
        else:
            status, color = "SAFE", "🟢"

        c1, c2, c3 = st.columns(3)
        c1.metric("Failure Probability", f"{failure_prob:.2f}")
        c2.metric("Predicted RUL", f"{rul_pred:.1f} minutes")
        c3.metric("Machine Status", f"{color} {status}")

        if warnings:
            st.warning(
                "⚠ Input exceeds training range for: "
                + ", ".join(warnings)
            )

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
with tab2:
    st.subheader("Feature Importance (RUL Model)")
    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": reg.feature_importances_
    }).sort_values("Importance")

    fig, ax = plt.subplots()
    ax.barh(imp["Feature"], imp["Importance"])
    st.pyplot(fig)
