import streamlit as st
import pandas as pd
import joblib

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="Online Payment Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------- LOAD MODEL ---------------------
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "model" / "fraud_detection_pipeline.pkl")

# --------------------- SIDEBAR ---------------------
st.sidebar.title("💳 Fraud Detection")

st.sidebar.markdown("""
## 📚 Project Details

*Project:* Online Payment Fraud Detection

*Algorithm:* Logistic Regression

*Dataset:* Online Payment Fraud Detection Dataset

*Machine Learning:* Scikit-Learn

*Deployment:* Streamlit

---
### 👩‍💻 Developed By

*Praveena Adabala*

Computer Science & Business Systems

---

### 📌 Purpose

Detect whether an online payment transaction is likely to be fraudulent using Machine Learning.
""")

# --------------------- MAIN TITLE ---------------------
st.title("💳 Online Payment Fraud Detection System")

st.markdown("""
Welcome to the *Online Payment Fraud Detection System*.

This application predicts whether a payment transaction is *Legitimate* or *Fraudulent*
using a trained Machine Learning model.
""")

with st.expander("📖 About this Project"):
    st.write("""
This application was developed using *Machine Learning* to detect fraudulent online payment transactions.

### Technologies Used
- Python
- Pandas
- Scikit-Learn
- Streamlit
- Joblib

### Model
Logistic Regression Pipeline

### Features
- User-friendly interface
- Real-time prediction
- Fraud probability
- Professional dashboard
""")

st.divider()

# --------------------- INPUTS ---------------------

col1, col2 = st.columns(2)

with col1:

    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
        help="Select the type of transaction."
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        help="Enter the transaction amount."
    )

    oldbalanceOrg = st.number_input(
        "Old Balance (Sender)",
        min_value=0.0,
        value=10000.0
    )

with col2:

    newbalanceOrig = st.number_input(
        "New Balance (Sender)",
        min_value=0.0,
        value=9000.0
    )

    oldbalanceDest = st.number_input(
        "Old Balance (Receiver)",
        min_value=0.0,
        value=0.0
    )

    newbalanceDest = st.number_input(
        "New Balance (Receiver)",
        min_value=0.0,
        value=0.0
    )

st.divider()

# --------------------- BUTTON ---------------------

if st.button("🔍 Predict Transaction", use_container_width=True):

    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_data)[0]

    st.divider()

    # ---------------- Probability ----------------

    probability_available = hasattr(model, "predict_proba")

    if probability_available:
        probability = model.predict_proba(input_data)[0][1]
    else:
        probability = None

    # ---------------- Prediction ----------------

    if prediction == 1:

        st.error("## 🚨 Fraudulent Transaction Detected")

        st.warning(
            "This transaction appears to be *high risk* and may be fraudulent."
        )

    else:

        st.success("## ✅ Legitimate Transaction")

        st.info(
            "This transaction appears to be *safe* based on the trained model."
        )

        st.balloons()

    # ---------------- Metrics ----------------

    if probability is not None:

        st.subheader("Prediction Confidence")

        st.metric(
            label="Fraud Probability",
            value=f"{probability:.2%}"
        )

        st.progress(float(probability))

    # ---------------- Input Summary ----------------

    st.subheader("Transaction Summary")

    st.dataframe(
        input_data,
        use_container_width=True
    )

# --------------------- FOOTER ---------------------

st.divider()

st.markdown("""
<center>

### 💳 Online Payment Fraud Detection System

Developed by *Praveena Adabala*

Computer Science & Business Systems

Machine Learning Mini Project • 2026

</center>
""", unsafe_allow_html=True)