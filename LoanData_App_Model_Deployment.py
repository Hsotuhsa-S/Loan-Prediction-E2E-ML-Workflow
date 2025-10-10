# LoanData_App_Model_Deployment.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, pickle
import os

# =============================
# Configuration
# =============================
# Update these paths to match your actual project structure
MODEL_PATH = "model/loanApproval/Loan_RandomForest_model.pkl"
DATA_PATH = "data/loanApproval/cleanLoanApprovalData.csv"

FEATURES = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]

TARGET = "loan_status"
LABEL_MAP = {0: "Rejected", 1: "Approved"}  # human-readable labels

# =============================
# Helpers
# =============================
@st.cache_resource
def load_model(path=MODEL_PATH):
    """Load trained pipeline (preprocessor + model)."""
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.info("Please ensure the model file exists in the correct location.")
        return None
    
    try:
        return joblib.load(path)
    except Exception as e:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            st.error(f"Failed to load model: {e2}")
            return None

@st.cache_data
def load_sample_data(path=DATA_PATH):
    """Load sample data for dropdown choices."""
    if not os.path.exists(path):
        st.warning(f"Sample data file not found at: {path}")
        st.info("Manual entry will use default ranges instead of data-driven choices.")
        return None
    
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load sample data: {e}")
        return None

def predict_df(model, df):
    """Make predictions on a DataFrame of inputs."""
    X = df[FEATURES].copy()
    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)
    except Exception:
        probs = None
    return preds, probs

# =============================
# App
# =============================
st.set_page_config(page_title="Loan Approval Classifier", page_icon="🏦")
st.title("🏦 Loan Approval Classifier")

st.write("Predict whether a loan application will be **Approved** or **Rejected** "
         "based on 12 applicant and loan features.")

# Load trained model
model = load_model()
if model is None:
    st.stop()

# Load dataset (for dropdown choices in manual entry)
df_sample = load_sample_data()

# Input mode
mode = st.radio("Choose input method:", ["Manual entry", "CSV upload"])

# =============================
# Manual entry
# =============================
if mode == "Manual entry":
    input_data = {}
    
    # Personal Information Section
    st.markdown("### 👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data["person_age"] = st.number_input("Age", min_value=18, max_value=100, value=30)
        input_data["person_gender"] = st.selectbox("Gender", ["Male", "Female"])
    
    with col2:
        input_data["person_education"] = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        input_data["person_income"] = st.number_input("Annual Income ($)", min_value=0, value=50000)
    
    with col3:
        input_data["person_emp_exp"] = st.number_input("Employment Experience (years)", min_value=0, value=5)
        input_data["person_home_ownership"] = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    
    # Loan Details Section
    st.markdown("### 💰 Loan Details")
    col1, col2 = st.columns(2)
    
    with col1:
        input_data["loan_amnt"] = st.number_input("Loan Amount ($)", min_value=0, value=10000)
        input_data["loan_intent"] = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"])
    
    with col2:
        input_data["loan_int_rate"] = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
        input_data["loan_percent_income"] = st.slider("Loan as % of Income", 0.0, 1.0, 0.2)
    
    # Credit Information Section
    st.markdown("### 📊 Credit Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data["credit_score"] = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    
    with col2:
        input_data["cb_person_cred_hist_length"] = st.number_input("Credit History Length (years)", min_value=0, value=10)
    
    with col3:
        input_data["previous_loan_defaults_on_file"] = st.selectbox("Previous Defaults", ["No", "Yes"])
    
    # Prediction button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🤖 Predict Loan Approval", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            preds, probs = predict_df(model, input_df)
            label = LABEL_MAP.get(preds[0], preds[0])
            st.success(f"Prediction: {label}")
            if probs is not None:
                st.write("Class probabilities:",
                         {LABEL_MAP.get(i, i): float(p) for i, p in enumerate(probs[0])})

# =============================
# CSV upload
# =============================
elif mode == "CSV upload":
    st.subheader("Upload CSV file with loan applications")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_input = pd.read_csv(uploaded)

        # Drop target column if present
        if TARGET in df_input.columns:
            df_input = df_input.drop(columns=[TARGET])

        # Check columns
        missing = [f for f in FEATURES if f not in df_input.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if st.button("Predict on CSV"):
                preds, probs = predict_df(model, df_input)
                df_input["prediction"] = [LABEL_MAP.get(p, p) for p in preds]
                if probs is not None:
                    df_input["pred_prob"] = probs.max(axis=1)
                st.write("Predictions (first 20 rows):")
                st.dataframe(df_input.head(20))

                # Allow download
                csv_bytes = df_input.to_csv(index=False).encode()
                st.download_button(
                    "Download predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
