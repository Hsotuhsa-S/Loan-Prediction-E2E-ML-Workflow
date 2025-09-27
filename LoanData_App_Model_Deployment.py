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
    st.subheader("Enter loan application details")
    
    # Group features into logical categories
    personal_features = ["person_age", "person_gender", "person_education", 
                         "person_income", "person_emp_exp", "person_home_ownership"]
    loan_features = ["loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income"]
    credit_features = ["cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"]
    
    input_data = {}
    
    # Personal Information Section
    st.markdown("### 👤 Personal Information")
    personal_cols = st.columns(3)
    for i, feature in enumerate(personal_features):
        with personal_cols[i % 3]:
            if df_sample is not None and df_sample[feature].dtype == "object":  # categorical
                choices = df_sample[feature].dropna().unique().tolist()
                input_data[feature] = st.selectbox(feature, choices)
            elif df_sample is not None:  # numeric with sample data
                default_val = float(df_sample[feature].median())
                input_data[feature] = st.number_input(feature, value=default_val)
            else:  # fallback when no sample data
                if feature in ["person_gender", "person_education", "person_home_ownership"]:
                    fallback_choices = {
                        "person_gender": ["Male", "Female"],
                        "person_education": ["High School", "Bachelor", "Master", "PhD"],
                        "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
                    }
                    input_data[feature] = st.selectbox(feature, fallback_choices.get(feature, ["Option1", "Option2"]))
                else:
                    fallback_defaults = {
                        "person_age": 30.0, "person_income": 50000.0, "person_emp_exp": 5.0,
                    }
                    default_val = fallback_defaults.get(feature, 0.0)
                    input_data[feature] = st.number_input(feature, value=default_val)
    
    # Loan Information Section
    st.markdown("### 💰 Loan Information")
    loan_cols = st.columns(2)
    for i, feature in enumerate(loan_features):
        with loan_cols[i % 2]:
            if df_sample is not None and df_sample[feature].dtype == "object":  # categorical
                choices = df_sample[feature].dropna().unique().tolist()
                input_data[feature] = st.selectbox(feature, choices)
            elif df_sample is not None:  # numeric with sample data
                default_val = float(df_sample[feature].median())
                input_data[feature] = st.number_input(feature, value=default_val)
            else:  # fallback when no sample data
                if feature == "loan_intent":
                    fallback_choices = {
                        "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                    }
                    input_data[feature] = st.selectbox(feature, fallback_choices.get(feature, ["Option1", "Option2"]))
                else:
                    fallback_defaults = {
                        "loan_amnt": 10000.0, "loan_int_rate": 10.0, "loan_percent_income": 0.2,
                    }
                    default_val = fallback_defaults.get(feature, 0.0)
                    input_data[feature] = st.number_input(feature, value=default_val)
    
    # Credit Information Section
    st.markdown("### 📊 Credit Information")
    credit_cols = st.columns(3)
    for i, feature in enumerate(credit_features):
        with credit_cols[i % 3]:
            if df_sample is not None and df_sample[feature].dtype == "object":  # categorical
                choices = df_sample[feature].dropna().unique().tolist()
                input_data[feature] = st.selectbox(feature, choices)
            elif df_sample is not None:  # numeric with sample data
                default_val = float(df_sample[feature].median())
                input_data[feature] = st.number_input(feature, value=default_val)
            else:  # fallback when no sample data
                if feature == "previous_loan_defaults_on_file":
                    fallback_choices = {
                        "previous_loan_defaults_on_file": ["Yes", "No"]
                    }
                    input_data[feature] = st.selectbox(feature, fallback_choices.get(feature, ["Option1", "Option2"]))
                else:
                    fallback_defaults = {
                        "cb_person_cred_hist_length": 10.0, "credit_score": 650.0
                    }
                    default_val = fallback_defaults.get(feature, 0.0)
                    input_data[feature] = st.number_input(feature, value=default_val)
    
    # Prediction button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predict", use_container_width=True):
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
