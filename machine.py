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

numeric_df = df.select_dtypes(include=np.number)
X = numeric_df.drop(columns=[failure_col], errors="ignore")
X = X.fillna(X.median())

# --------------------------------------------------
# CREATE RUL
# --------------------------------------------------
df["Operating_Time"] = np.arange(len(df))
df["RUL"] = df["Operating_Time"].max() - df["Operating_Time"]

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

y_rul = df["RUL"]
Xtr, Xte, ytr, yte = train_test_split(X, y_rul, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(Xtr, ytr)

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
tab1, tab2 = st.tabs(["📊 Prediction", "📈 Feature Importance"])

# ==================================================
# TAB 1 : INPUT + OUTPUT (MAIN SCREEN)
# ==================================================
with tab1:

    st.subheader("🧮 Enter Machine Sensor Values")

    with st.form("main_input_form"):
        col1, col2, col3 = st.columns(3)
        user_inputs = {}

        cols = list(X.columns)

        for i, feature in enumerate(cols):
            if i % 3 == 0:
                user_inputs[feature] = col1.number_input(
                    feature, value=float(df[feature].median())
                )
            elif i % 3 == 1:
                user_inputs[feature] = col2.number_input(
                    feature, value=float(df[feature].median())
                )
            else:
                user_inputs[feature] = col3.number_input(
                    feature, value=float(df[feature].median())
                )

        predict_btn = st.form_submit_button("🚀 Predict Machine Health")

    # --------------------------------------------------
    # PREDICTION OUTPUT
    # --------------------------------------------------
    if predict_btn:
        input_df = pd.DataFrame([user_inputs])

        failure_prob = clf.predict_proba(input_df)[0][1] if clf else 0
        rul_pred = max(reg.predict(input_df)[0], 0)

        if failure_prob > 0.7 or rul_pred < 30:
            status = "🔴 HIGH RISK"
        elif failure_prob > 0.4 or rul_pred < 100:
            status = "🟠 MEDIUM RISK"
        else:
            status = "🟢 SAFE"

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Failure Probability", f"{failure_prob:.2f}")
        c2.metric("Remaining Useful Life", f"{rul_pred:.1f} minutes")
        c3.metric("Machine Status", status)

# ==================================================
# TAB 2 : FEATURE IMPORTANCE
# ==================================================
with tab2:
    st.subheader("📈 Feature Importance (RUL Model)")

    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": reg.feature_importances_
    }).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.set_xlabel("Importance Score")
    ax.set_title("Key Factors Affecting RUL")
    st.pyplot(fig)
