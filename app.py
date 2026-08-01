import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load(open("Telco Customer Churn.pkl", "rb"))
scaler = joblib.load(open("scaler.pkl", "rb"))

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

h1{
    color:#1f77b4;
}

.stButton>button{
    background-color:#1f77b4;
    color:white;
    border-radius:8px;
    width:100%;
    height:50px;
    font-size:18px;
}

.result{
    padding:20px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction")

st.write(
"""
Predict whether a customer is likely to churn using Machine Learning.
"""
)

st.sidebar.title("Customer Churn")

st.sidebar.info(
"""
This application predicts whether a customer is likely to churn.

Model:
- Logistic Regression

Preprocessing:
- StandardScaler

Dataset:
IBM Telco Customer Churn
"""
)

# -----------------------------
# INPUTS
# -----------------------------

col1,col2=st.columns(2)

with col1:

    gender=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    senior=st.selectbox(
        "Senior Citizen",
        ["No","Yes"]
    )

    partner=st.selectbox(
        "Partner",
        ["No","Yes"]
    )

    dependents=st.selectbox(
        "Dependents",
        ["No","Yes"]
    )

    tenure=st.slider(
        "Tenure",
        0,
        72,
        12
    )

    phone=st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )

    multiple=st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet=st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    security=st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    backup=st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with col2:

    protection=st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    support=st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv=st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies=st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    paperless=st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
    )

    contract=st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    payment=st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

    monthly=st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total=st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

predict=st.button("Predict Customer Churn")
if predict:

    data = {

        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,

        "tenure": tenure,

        "PhoneService": 1 if phone == "Yes" else 0,

        "PaperlessBilling": 1 if paperless == "Yes" else 0,

        "MonthlyCharges": monthly,

        "TotalCharges": total,

        # Multiple Lines
        "MultipleLines_No phone service": 1 if multiple == "No phone service" else 0,
        "MultipleLines_Yes": 1 if multiple == "Yes" else 0,

        # Internet Service (DSL is reference)
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No": 1 if internet == "No" else 0,

        # Online Security
        "OnlineSecurity_No internet service": 1 if security == "No internet service" else 0,
        "OnlineSecurity_Yes": 1 if security == "Yes" else 0,

        # Online Backup
        "OnlineBackup_No internet service": 1 if backup == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if backup == "Yes" else 0,

        # Device Protection
        "DeviceProtection_No internet service": 1 if protection == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if protection == "Yes" else 0,

        # Tech Support
        "TechSupport_No internet service": 1 if support == "No internet service" else 0,
        "TechSupport_Yes": 1 if support == "Yes" else 0,

        # Streaming TV
        "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,

        # Streaming Movies
        "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,

        # Contract (Month-to-month is reference)
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,

        # Payment Method (Bank transfer is reference)
        "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([data])

    feature_order = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "PaperlessBilling",
        "MonthlyCharges",
        "TotalCharges",
        "MultipleLines_No phone service",
        "MultipleLines_Yes",
        "InternetService_Fiber optic",
        "InternetService_No",
        "OnlineSecurity_No internet service",
        "OnlineSecurity_Yes",
        "OnlineBackup_No internet service",
        "OnlineBackup_Yes",
        "DeviceProtection_No internet service",
        "DeviceProtection_Yes",
        "TechSupport_No internet service",
        "TechSupport_Yes",
        "StreamingTV_No internet service",
        "StreamingTV_Yes",
        "StreamingMovies_No internet service",
        "StreamingMovies_Yes",
        "Contract_One year",
        "Contract_Two year",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check"
    ]

    input_df = input_df[feature_order]

    scaled_data = scaler.transform(input_df)
    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🚨 Customer is likely to Churn")

        st.progress(float(probability))

        st.metric(
            label="Churn Probability",
            value=f"{probability*100:.2f}%"
        )

        st.warning("""
### Recommended Retention Strategies

- 📞 Contact the customer immediately.
- 💰 Offer a discount or loyalty plan.
- 📦 Recommend a better subscription package.
- ⭐ Provide priority customer support.
- 🎁 Offer exclusive benefits.
""")

    else:

        st.success("✅ Customer is likely to Stay")

        st.progress(float(1-probability))

        st.metric(
            label="Customer Retention Probability",
            value=f"{(1-probability)*100:.2f}%"
        )

        st.info("""
### Recommendation

- 😊 Maintain good customer service.
- 💬 Continue regular engagement.
- 🎯 Recommend suitable upgrades.
- ❤️ Reward loyal customers.
""")

    st.markdown("---")

    st.subheader("Customer Information")

    st.dataframe(input_df, use_container_width=True)

    st.markdown("---")

    st.caption("Developed using Streamlit • Scikit-Learn • Logistic Regression")