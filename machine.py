import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ======================================================
# PAGE CONFIG + CUSTOM CSS
# ======================================================
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    text-align: center;
}
.safe {color: #2ecc71; font-weight: bold;}
.medium {color: #f39c12; font-weight: bold;}
.high {color: #e74c3c; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
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
failure_candidates = ["Machine failure", "Failure", "Target", "Label"]
rul_candidates = ["RUL", "Remaining Useful Life"]

failure_col = next((c for c in failure_candidates if c in df.columns), None)
rul_col = next((c for c in rul_candidates if c in df.columns), None)

# ======================================================
# FEATURE SELECTION
# ======================================================
numeric_df = df.select_dtypes(include=[np.number])

drop_cols = []
if failure_col: drop_cols.append(failure_col)
if rul_col: drop_cols.append(rul_col)

X = numeric_df.drop(columns=drop_cols, errors="ignore")
X = X.fillna(X.median())

# ======================================================
# ADD OPERATING TIME
# ======================================================
if "Operating_Time" not in X.columns:
    X["Operating_Time"] = np.arange(len(X))
    df["Operating_Time"] = X["Operating_Time"]

# ======================================================
# TRAIN MODELS
# ======================================================
rf_clf = None
if failure_col:
    y_fail = df[failure_col]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_fail, test_size=0.2, random_state=42, stratify=y_fail
    )
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_clf.fit(Xtr, ytr)

if rul_col is None:
    df["RUL"] = df["Operating_Time"].max() - df["Operating_Time"]
    rul_col = "RUL"

y_rul = df[rul_col]
Xtr, Xte, ytr, yte = train_test_split(
    X, y_rul, test_size=0.2, random_state=42
)
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(Xtr, ytr)

# ======================================================
# INPUT SAFETY
# ======================================================
def clip_input(val, col):
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    exceeded = val < lo or val > hi
    return np.clip(val, lo, hi), exceeded

def build_input(user_inputs):
    row, warns = {}, []
    for col in X.columns:
        if col in user_inputs:
            v, ex = clip_input(user_inputs[col], col)
            row[col] = v
            if ex: warns.append(col)
        else:
            row[col] = df[col].median()
    return pd.DataFrame([row]), warns

# ======================================================
# SIDEBAR INPUT
# ======================================================
st.sidebar.header("⚙ Machine Parameters")

user_inputs = {}
for col in X.columns:
    user_inputs[col] = st.sidebar.number_input(
        col, value=float(df[col].median())
    )

# ======================================================
# MAIN TABS
# ======================================================
tab1, tab2 = st.tabs(["📊 Prediction", "📈 Feature Importance"])

# ======================================================
# PREDICTION TAB
# ======================================================
with tab1:
    if st.button("🚀 Predict Machine Health", use_container_width=True):

        input_df, warn_cols = build_input(user_inputs)

        failure_prob = None
        if rf_clf:
            failure_prob = rf_clf.predict_proba(input_df)[0][1]

        rul_pred = rf_reg.predict(input_df)[0]
        rul_minutes = max(rul_pred, 0)

        if failure_prob is not None:
            if failure_prob >= 0.7 or rul_minutes < 30:
                status, css = "HIGH RISK", "high"
                progress = 90
            elif failure_prob >= 0.4 or rul_minutes < 100:
                status, css = "MEDIUM RISK", "medium"
                progress = 60
            else:
                status, css = "SAFE", "safe"
                progress = 25
        else:
            status, css = "SAFE", "safe"
            progress = 25

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div class='metric-card'><h3>Failure Probability</h3><h2>{failure_prob:.2f}</h2></div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='metric-card'><h3>Predicted RUL</h3><h2>{rul_minutes:.1f} min</h2></div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div class='metric-card'><h3>Status</h3><h2 class='{css}'>{status}</h2></div>", unsafe_allow_html=True)

        st.progress(progress)

        if warn_cols:
            st.warning(
                f"⚠ Input exceeds training range for: {', '.join(warn_cols)}. "
                "Values were safely adjusted."
            )

# ======================================================
# FEATURE IMPORTANCE TAB
# ======================================================
with tab2:
    st.subheader("Feature Importance (RUL Model)")

    fig, ax = plt.subplots(figsize=(7,4))
    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_reg.feature_importances_
    }).sort_values(by="Importance")

    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.set_xlabel("Importance Score")
    ax.set_title("Key Factors Affecting RUL")
    st.pyplot(fig)
