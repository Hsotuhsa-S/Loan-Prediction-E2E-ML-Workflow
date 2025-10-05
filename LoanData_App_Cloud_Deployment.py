# LoanData_App_Cloud_Deployment.py
"""
Streamlit Cloud Optimized Loan Approval Prediction App

This version is specifically optimized for Streamlit Cloud deployment,
addressing common issues like:
- Library version compatibility
- Memory constraints
- File path handling
- Pickle loading issues in cloud environments
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import warnings
from typing import Optional, Tuple, Dict, Any

# Early compatibility checks
def check_environment():
    """Check and display environment information for debugging"""
    env_info = {
        "Python Version": sys.version,
        "Platform": sys.platform,
        "Working Directory": os.getcwd(),
        "Available Memory": "Unknown"
    }
    
    try:
        import psutil
        env_info["Available Memory"] = f"{psutil.virtual_memory().available / (1024**3):.1f} GB"
    except ImportError:
        pass
    
    return env_info

# Import ML libraries with error handling
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError as e:
    st.error(f"joblib import failed: {e}")
    JOBLIB_AVAILABLE = False

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError as e:
    st.error(f"pickle import failed: {e}")
    PICKLE_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
    SKLEARN_VERSION = sklearn.__version__
except ImportError as e:
    st.error(f"scikit-learn import failed: {e}")
    SKLEARN_AVAILABLE = False
    SKLEARN_VERSION = "Unknown"

# =============================
# Configuration
# =============================
MODEL_PATH = "model/loanApproval/Loan_RandomForest_model.pkl"
DATA_PATH = "data/loanApproval/cleanLoanApprovalData.csv"

FEATURES = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file"
]

TARGET = "loan_status"
LABEL_MAP = {0: "❌ Rejected", 1: "✅ Approved"}

# =============================
# Cloud-Optimized Model Loading
# =============================

@st.cache_resource
def load_model_cloud_optimized(path: str = MODEL_PATH) -> Optional[Any]:
    """
    Cloud-optimized model loading with comprehensive error handling.
    Specifically designed for Streamlit Cloud deployment.
    """
    loading_log = []
    
    # Environment check
    env_info = check_environment()
    loading_log.append(f"Environment: {env_info['Platform']}")
    loading_log.append(f"Python: {env_info['Python Version'][:20]}...")
    
    # File existence and path checks
    if not os.path.exists(path):
        # Try alternative paths (cloud vs local)
        alternative_paths = [
            path,
            f"./{path}",
            f"/{path}",
            path.replace("\\", "/"),  # Windows to Unix path
            path.replace("/", "\\")   # Unix to Windows path
        ]
        
        found_path = None
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                found_path = alt_path
                break
        
        if not found_path:
            loading_log.append(f"❌ Model file not found at any of: {alternative_paths}")
            st.error("🚫 Model file not found")
            with st.expander("🔍 Debug Information"):
                st.write("\n".join(loading_log))
                st.write(f"**Available files in model directory:**")
                try:
                    if os.path.exists("model"):
                        for root, dirs, files in os.walk("model"):
                            for file in files:
                                st.write(f"- {os.path.join(root, file)}")
                except Exception as e:
                    st.write(f"Error listing files: {e}")
            return None
        else:
            path = found_path
            loading_log.append(f"✅ Found model at: {path}")
    
    # File size check
    try:
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        loading_log.append(f"📁 File size: {file_size_mb:.1f} MB")
        
        # Warn if file is very large for cloud deployment
        if file_size_mb > 100:
            st.warning(f"⚠️ Large model file ({file_size_mb:.1f} MB) may cause memory issues in cloud")
    except Exception as e:
        loading_log.append(f"⚠️ Could not check file size: {e}")
    
    # Library availability check
    if not JOBLIB_AVAILABLE and not PICKLE_AVAILABLE:
        st.error("❌ Neither joblib nor pickle is available")
        return None
    
    loading_log.append(f"📦 Scikit-learn version: {SKLEARN_VERSION}")
    loading_log.append(f"📦 Joblib available: {JOBLIB_AVAILABLE}")
    loading_log.append(f"📦 Pickle available: {PICKLE_AVAILABLE}")
    
    # Try multiple loading strategies optimized for cloud
    loading_strategies = []
    
    if JOBLIB_AVAILABLE:
        loading_strategies.extend([
            ("joblib.load (default)", lambda p: joblib.load(p)),
            ("joblib.load (no mmap)", lambda p: joblib.load(p, mmap_mode=None)),
        ])
    
    if PICKLE_AVAILABLE:
        loading_strategies.extend([
            ("pickle.load", lambda p: pickle.load(open(p, "rb")))
        ])
    
    # Try alternative pickle libraries if available
    try:
        import pickle5
        loading_strategies.append(("pickle5.load", lambda p: pickle5.load(open(p, "rb"))))
        loading_log.append("📦 pickle5 available")
    except ImportError:
        loading_log.append("📦 pickle5 not available")
    
    try:
        import cloudpickle
        loading_strategies.append(("cloudpickle.load", lambda p: cloudpickle.load(open(p, "rb"))))
        loading_log.append("📦 cloudpickle available")
    except ImportError:
        loading_log.append("📦 cloudpickle not available")
    
    # Attempt loading with each strategy
    model = None
    successful_method = None
    
    for method_name, load_func in loading_strategies:
        try:
            loading_log.append(f"⏳ Trying {method_name}...")
            
            # Set warnings and memory optimizations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Try loading with memory optimization
                model = load_func(path)
            
            # Validate loaded model
            if model is not None and hasattr(model, 'predict'):
                loading_log.append(f"✅ SUCCESS with {method_name}")
                successful_method = method_name
                break
            else:
                loading_log.append(f"⚠️ {method_name}: Invalid model object")
                model = None
                
        except Exception as e:
            error_msg = str(e)
            loading_log.append(f"❌ {method_name}: {error_msg}")
            
            # Specific error handling for cloud issues
            if "STACK_GLOBAL" in error_msg:
                loading_log.append("   → This is a pickle compatibility issue")
            elif "memory" in error_msg.lower():
                loading_log.append("   → This might be a memory constraint issue")
            elif "version" in error_msg.lower():
                loading_log.append("   → This might be a version compatibility issue")
            
            continue
    
    # Display results
    if model is not None:
        st.success(f"✅ Model loaded successfully using {successful_method}")
        
        # Display model info
        try:
            model_info = {
                "Type": type(model).__name__,
                "Features": getattr(model, 'n_features_in_', 'Unknown'),
                "Sklearn Version": getattr(model, '_sklearn_version', 'Unknown')
            }
            
            if hasattr(model, 'steps'):
                model_info["Pipeline Steps"] = len(model.steps)
            
            with st.expander("📊 Model Information", expanded=False):
                for key, value in model_info.items():
                    st.write(f"**{key}:** {value}")
        except Exception as e:
            loading_log.append(f"⚠️ Could not extract model info: {e}")
    else:
        st.error("❌ Failed to load model with any available method")
        
        # Provide troubleshooting information
        with st.expander("🔧 Troubleshooting Information", expanded=True):
            st.write("**Loading attempts:**")
            for log_entry in loading_log:
                st.write(log_entry)
            
            st.write("\n**Possible solutions:**")
            st.write("1. Check if model file exists in the repository")
            st.write("2. Verify scikit-learn version compatibility")
            st.write("3. Try re-saving the model with current library versions")
            st.write("4. Check Streamlit Cloud memory limits")
    
    # Always show debug info in expander for troubleshooting
    with st.expander("🔍 Debug Log", expanded=False):
        for log_entry in loading_log:
            st.write(log_entry)
    
    return model


@st.cache_data
def load_sample_data_cloud(path: str = DATA_PATH) -> Optional[pd.DataFrame]:
    """Cloud-optimized data loading"""
    if not os.path.exists(path):
        # Try alternative paths
        alt_paths = [path, f"./{path}", path.replace("\\", "/")]
        found_path = None
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                found_path = alt_path
                break
        
        if not found_path:
            st.info(f"📝 Sample data not found. Using fallback options.")
            return None
        path = found_path
    
    try:
        df = pd.read_csv(path)
        st.info(f"📊 Sample data loaded: {len(df)} records")
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load sample data: {e}")
        return None


def predict_with_error_handling(model, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Make predictions with comprehensive cloud-specific error handling"""
    try:
        X = df[FEATURES].copy()
        
        # Basic prediction
        predictions = model.predict(X)
        
        # Try to get probabilities
        probabilities = None
        try:
            probabilities = model.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)
        except Exception as e:
            st.warning(f"⚠️ Could not get prediction probabilities: {e}")
            confidence_scores = np.ones(len(predictions)) * 0.5
        
        return predictions, probabilities, confidence_scores
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        raise e


# =============================
# Fallback Options
# =============================

def get_fallback_choices() -> Dict[str, list]:
    """Provide fallback choices when sample data is unavailable"""
    return {
        "person_gender": ["Male", "Female"],
        "person_education": ["High School", "Bachelor", "Master", "PhD"],
        "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
        "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", 
                       "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        "previous_loan_defaults_on_file": ["No", "Yes"]
    }


def get_fallback_defaults() -> Dict[str, float]:
    """Provide fallback numeric defaults"""
    return {
        "person_age": 30.0, "person_income": 50000.0, "person_emp_exp": 5.0,
        "loan_amnt": 10000.0, "loan_int_rate": 10.0, "loan_percent_income": 0.2,
        "cb_person_cred_hist_length": 10.0, "credit_score": 650.0
    }


# =============================
# Main Application
# =============================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Loan Approval Classifier - Cloud",
        page_icon="🏦",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏦 Loan Approval Classifier")
    st.caption("Streamlit Cloud Optimized Version")
    
    # Environment information in sidebar
    with st.sidebar:
        st.header("🌐 Environment Info")
        env_info = check_environment()
        st.write(f"**Platform:** {env_info['Platform']}")
        st.write(f"**Python:** {env_info['Python Version'][:10]}...")
        st.write(f"**Scikit-learn:** {SKLEARN_VERSION}")
        
        if st.button("🔄 Refresh Environment"):
            st.cache_resource.clear()
            st.rerun()
    
    # Load model and data
    st.header("🔧 System Status")
    
    with st.spinner("Loading model and data..."):
        model = load_model_cloud_optimized()
        df_sample = load_sample_data_cloud()
    
    if model is None:
        st.error("🚫 Cannot proceed without a valid model")
        st.stop()
    
    st.success("🎉 System ready for predictions!")
    
    # Simple prediction interface (optimized for cloud)
    st.header("📝 Loan Application")
    
    # Personal Information Section
    st.markdown("### 👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        person_education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    with col3:
        person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, value=5)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    # Loan Information Section
    st.markdown("### 💰 Loan Details")
    loan_col1, loan_col2 = st.columns(2)
    
    with loan_col1:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
        loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE"])
    with loan_col2:
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)
        loan_percent_income = st.slider("Loan as % of Income", 0.0, 1.0, 0.2)

    # Credit Information Section
    st.markdown("### 📊 Credit Information")
    credit_col1, credit_col2, credit_col3 = st.columns(3)
    
    with credit_col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    with credit_col2:
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=10)
    with credit_col3:
        previous_loan_defaults_on_file = st.selectbox("Previous Defaults", ["No", "Yes"])
    
    # Prediction
    if st.button("🔮 Predict Loan Approval", type="primary"):
        try:
            # Create input dataframe
            input_data = {
                "person_age": person_age,
                "person_gender": person_gender,
                "person_education": person_education,
                "person_income": person_income,
                "person_emp_exp": person_emp_exp,
                "person_home_ownership": person_home_ownership,
                "loan_amnt": loan_amnt,
                "loan_intent": loan_intent,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
                "credit_score": credit_score,
                "previous_loan_defaults_on_file": previous_loan_defaults_on_file
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            with st.spinner("Making prediction..."):
                predictions, probabilities, confidence_scores = predict_with_error_handling(model, input_df)
            
            # Display result
            result = predictions[0]
            confidence = confidence_scores[0]
            label = LABEL_MAP.get(result, f"Unknown ({result})")
            
            if result == 1:
                st.success(f"## {label}")
                st.balloons()
            else:
                st.error(f"## {label}")
            
            # Show confidence
            st.metric("Confidence", f"{confidence:.1%}")
            
            # Show probabilities if available
            if probabilities is not None:
                st.write("**Probabilities:**")
                st.write(f"• Rejected: {probabilities[0][0]:.1%}")
                st.write(f"• Approved: {probabilities[0][1]:.1%}")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please check your input values and try again.")


if __name__ == "__main__":
    main()