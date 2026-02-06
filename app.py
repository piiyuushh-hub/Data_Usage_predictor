import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="Telecom Data Usage Predictor",
    page_icon="📶",
    layout="wide"
)

# Load saved files
model = pickle.load(open("linear_regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Title section
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>📊 Telecom Monthly Data Usage Predictor</h1>
    <p style='text-align: center;'>Predict customer internet usage using Machine Learning</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("⚙️ Customer Profile")
st.sidebar.info("Enter customer details to estimate monthly data usage")

# Sidebar inputs
customer_age = st.sidebar.slider("Customer Age", 18, 80, 30)
tenure_months = st.sidebar.slider("Tenure (Months)", 1, 120, 12)
plan_type = st.sidebar.selectbox("Plan Type", ["Prepaid", "Postpaid"])
device_type = st.sidebar.selectbox("Device Type", ["Android", "iOS", "Other"])
network_type = st.sidebar.selectbox("Network Type", ["3G", "4G", "5G"])
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
payment_method = st.sidebar.selectbox("Payment Method", ["UPI", "Card", "Cash"])

# Main layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📞 Usage Details")
    monthly_recharge = st.number_input("Monthly Recharge (₹)", 100, 5000, 499)
    call_minutes = st.number_input("Call Minutes", 0, 3000, 300)
    sms_count = st.number_input("SMS Count", 0, 1000, 50)
    support_calls = st.number_input("Support Calls", 0, 20, 1)

with col2:
    st.subheader("🌐 Internet Details")
    internet_speed_mbps = st.selectbox("Internet Speed (Mbps)", [10, 20, 40, 100, 200])
    roaming_usage_gb = st.number_input("Roaming Usage (GB)", 0.0, 50.0, 1.0)

# Prepare input data
input_data = pd.DataFrame([{
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
    "region": region,
    "payment_method": payment_method
}])

# Encoding & scaling
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=columns, fill_value=0)
input_scaled = scaler.transform(input_data)

# Prediction button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Predict Monthly Data Usage"):
    prediction = model.predict(input_scaled)[0]

    # Prediction card
    st.markdown(
        f"""
        <div style='background-color:#D6EAF8;padding:20px;border-radius:10px'>
        <h3 style='color:#1B4F72'>📶 Predicted Monthly Data Usage</h3>
        <h2 style='color:#117A65'>{prediction:.2f} GB</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Visual Insights")

    # -------- GRAPH 1: Usage Drivers (Bar Chart) --------
    usage_drivers = pd.DataFrame({
        "Factor": ["Monthly Recharge", "Call Minutes", "Internet Speed", "Roaming Usage"],
        "Value": [
            monthly_recharge / 100,
            call_minutes / 50,
            internet_speed_mbps,
            roaming_usage_gb * 5
        ]
    })

    st.write("🔹 **Key Factors Influencing Data Usage**")
    st.bar_chart(usage_drivers.set_index("Factor"))

    # -------- GRAPH 2: Usage vs Recharge (Trend Line) --------
    recharge_range = np.linspace(100, monthly_recharge + 1000, 10)

    trend_df = pd.DataFrame({
        "Monthly Recharge": recharge_range,
        "Estimated Usage (GB)": recharge_range * 0.02
    })

    st.write("🔹 **Trend: Monthly Recharge vs Data Usage**")
    st.line_chart(trend_df.set_index("Monthly Recharge"))

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray'>
    Built using Machine Learning & Streamlit 🚀
    </p>
    """,
    unsafe_allow_html=True
)
