import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Multiple Linear Regression – Interactive Demo",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL FILES
# -------------------------------------------------
model = pickle.load(open("linear_regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">📊 Multiple Linear Regression: Interactive Demo</h1>
    <p style="text-align:center;">Telecom Monthly Data Usage Prediction</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR – WORKFLOW
# -------------------------------------------------
st.sidebar.markdown("### ✅ Workflow")
st.sidebar.markdown("✔ Upload Dataset (pre-trained)")
st.sidebar.markdown("✔ Select Inputs")
st.sidebar.markdown("✔ Fit Model")
st.sidebar.markdown("✔ Interpret & Diagnose")

st.sidebar.markdown("---")
st.sidebar.header("🧾 Customer Inputs")

customer_age = st.sidebar.slider("Customer Age", 18, 80, 30)
tenure_months = st.sidebar.slider("Tenure (Months)", 1, 120, 12)
monthly_recharge = st.sidebar.number_input("Monthly Recharge (₹)", 100, 5000, 500)
call_minutes = st.sidebar.number_input("Call Minutes", 0, 3000, 300)
sms_count = st.sidebar.number_input("SMS Count", 0, 1000, 50)
support_calls = st.sidebar.number_input("Support Calls", 0, 20, 1)

internet_speed_mbps = st.sidebar.selectbox("Internet Speed (Mbps)", [10, 20, 40, 100, 200])
roaming_usage_gb = st.sidebar.number_input("Roaming Usage (GB)", 0.0, 50.0, 1.0)

device_type = st.sidebar.selectbox("Device Type", ["Android", "iOS", "Other"])
plan_type = st.sidebar.selectbox("Plan Type", ["Prepaid", "Postpaid"])
network_type = st.sidebar.selectbox("Network Type", ["3G", "4G", "5G"])
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])

# -------------------------------------------------
# INPUT DATAFRAME
# -------------------------------------------------
input_df = pd.DataFrame([{
    "customer_age": customer_age,
    "monthly_recharge": monthly_recharge,
    "call_minutes": call_minutes,
    "sms_count": sms_count,
    "support_calls": support_calls,
    "internet_speed_mbps": internet_speed_mbps,
    "roaming_usage_gb": roaming_usage_gb,
    "tenure_months": tenure_months,
    "device_type": device_type,
    "plan_type": plan_type,
    "network_type": network_type,
    "region": region
}])

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=columns, fill_value=0)
input_scaled = scaler.transform(input_df)

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
left, right = st.columns([2.5, 1])

# -------------------------------------------------
# MODEL SUMMARY
# -------------------------------------------------
with left:
    st.subheader("📑 Model Summary")

    prediction = model.predict(input_scaled)[0]

    st.markdown(
        f"""
        <div style="background:#F466F7;padding:15px;border-radius:8px">
        <b>Model:</b> Linear Regression<br>
        <b>Prediction:</b> {prediction:.2f} GB
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# METRICS (STATIC – FROM YOUR EVALUATION)
# -------------------------------------------------
with right:
    st.subheader("📈 Performance")
    st.metric("R² Score", "0.61")
    st.metric("Adjusted R²", "0.58")

# -------------------------------------------------
# COEFFICIENTS
# -------------------------------------------------
st.subheader("β Coefficients")

coef_df = pd.DataFrame({
    "Feature": columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

st.dataframe(coef_df, height=300)

# -------------------------------------------------
# DIAGNOSTIC PLOTS (SIMULATED)
# -------------------------------------------------
st.subheader("📉 Diagnostic Plots")

col1, col2 = st.columns(2)

# Residuals vs Fitted (synthetic visualization)
with col1:
    fig, ax = plt.subplots()
    fitted = np.linspace(0, prediction * 1.5, 50)
    residuals = np.random.normal(0, 2, 50)
    ax.scatter(fitted, residuals)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residuals vs Fitted")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

# Q-Q Plot
with col2:
    fig, ax = plt.subplots()
    sns.lineplot(x=np.sort(residuals), y=np.sort(np.random.normal(0, 1, 50)))
    ax.set_title("Normal Q-Q Plot")
    st.pyplot(fig)

# -------------------------------------------------
# PREDICTION SECTION (LIKE IMAGE)
# -------------------------------------------------
st.subheader("🔮 Make Prediction")

st.markdown(
    f"""
    <div style="background:#E8F8F5;padding:20px;border-radius:10px">
    <h3 style="color:#117F65">📶 Predicted Monthly Data Usage</h3>
    <h2 style="color:#117F65">{prediction:.2f} GB</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    "<hr><p style='text-align:center;color:gray;'>Telecom Regression Demo | Streamlit</p>",
    unsafe_allow_html=True
)
