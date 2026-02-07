import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title("🔧 Predictive Maintenance for Industrial Machinery")
st.caption("Failure Prediction • RUL Estimation • Risk Assessment")

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    return pd.read_csv("ai4i2020.csv")

df = load_data()

# ======================================================
# TARGET DETECTION
# ======================================================
failure_col = "Machine failure" if "Machine failure" in df.columns else None

numeric_df = df.select_dtypes(include=np.number)
X = numeric_df.drop(columns=[failure_col], errors="ignore")
X = X.fillna(X.median())

# ======================================================
# CREATE RUL (IN MINUTES)
# ======================================================
df["Operating_Time"] = np.arange(len(df))
df["RUL"] = df["Operating_Time"].max() - df["Operating_Time"]

# ======================================================
# TRAIN MODELS
# ======================================================
if failure_col:
    y_fail = df[failure_col]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_fail, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
else:
    clf = None

y_rul = df["RUL"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y_rul, test_size=0.2, random_state=42
)
reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(Xtr, ytr)

# ======================================================
# RUL CONVERSION FUNCTION
# ======================================================
def format_rul(minutes):
    minutes = int(minutes)

    days = minutes // (24 * 60)
    hours = (minutes % (24 * 60)) // 60
    mins = minutes % 60

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    parts.append(f"{mins} minute{'s' if mins > 1 else ''}")

    return " ".join(parts)

# ======================================================
# MAIN TABS
# ======================================================
tab1, tab2 = st.tabs(["📊 Prediction", "📈 Feature Importance"])

# ======================================================
# TAB 1 — INPUT + OUTPUT
# ======================================================
with tab1:

    st.subheader("🧮 Enter Machine Sensor Values")

    with st.form("machine_input_form"):
        col1, col2, col3 = st.columns(3)
        user_inputs = {}

        features = list(X.columns)

        for i, feature in enumerate(features):
            default = float(df[feature].median())
            if i % 3 == 0:
                user_inputs[feature] = col1.number_input(feature, value=default)
            elif i % 3 == 1:
                user_inputs[feature] = col2.number_input(feature, value=default)
            else:
                user_inputs[feature] = col3.number_input(feature, value=default)

        submit = st.form_submit_button("🚀 Predict Machine Health")

    # ======================================================
    # PREDICTION OUTPUT
    # ======================================================
    if submit:
        input_df = pd.DataFrame([user_inputs])

        # Failure probability (percentage)
        failure_prob = clf.predict_proba(input_df)[0][1] if clf else 0
        failure_percent = failure_prob * 100

        # RUL prediction
        rul_minutes = max(reg.predict(input_df)[0], 0)
        rul_readable = format_rul(rul_minutes)

        # Risk logic
        if failure_percent > 70 or rul_minutes < 1440:
            status = "🔴 HIGH RISK"
        elif failure_percent > 40 or rul_minutes < 7200:
            status = "🟠 MEDIUM RISK"
        else:
            status = "🟢 SAFE"

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        c1.metric("Failure Probability", f"{failure_percent:.2f} %")
        c2.metric("Remaining Useful Life", rul_readable)
        c3.metric("Machine Status", status)

# ======================================================
# TAB 2 — FEATURE IMPORTANCE
# ======================================================
with tab2:

    st.subheader("📈 Feature Importance (RUL Model)")

    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": reg.feature_importances_
    }).sort_values(by="Importance")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.set_xlabel("Importance Score")
    ax.set_title("Key Factors Affecting RUL")
    st.pyplot(fig)
