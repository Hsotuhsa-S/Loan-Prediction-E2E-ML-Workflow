# Loan Approval Classification - End-to-End ML Workflow

**Machine learning project demonstrating complete data science pipeline from raw data to production-ready web application.**

## 🎯 Project Summary

**Objective**: Predict loan approval decisions using applicant and loan characteristics  
**Achievement**: Compared Random Forest vs Gradient Boosting. Developed and deployed a Random Forest classifier achieving **98% ROC-AUC** with interactive web interface

### Key Tasks:
- ✅ **Data Processing**: Cleaned 45000 loan records, handled missing values
- ✅ **Model Development**: Compared Random Forest vs Gradient Boosting with hyperparameter optimization  
- ✅ **Performance**: Achieved 98% ROC-AUC with robust cross-validation
- ✅ **Production Deployment**: Built interactive Streamlit web app for real-time predictions
- ✅ **Business Impact**: Previous loan defaults, loan interest rate and loan percent income as top predictors for loan decisions.
  The model reduces loan default risk by accurately identifying 89.2% of approved good customers, while capturing 77.5% of creditworthy customers, improving portfolio quality and minimizing losses.

## 📁 Project Structure

```
00_ClassificationModels-python/
├── 00_LoanData_Cleanup_EDA.ipynb          # Data cleaning and exploratory analysis
├── 01_modelTuningSelectionEvaluation.ipynb # Model training and evaluation
├── LoanData_App_Model_Deployment.py       # Streamlit web application
├── requirements.txt                        # Project dependencies
├── README.md                              # Project documentation
├── data/
│   └── loanApproval/
│       ├── loan_data.csv                  # Raw dataset
│       ├── cleanLoanApprovalData.csv      # Cleaned dataset
│       
├── model/
│   └── loanApproval/
│       └── Loan_RandomForest_model.pkl    # Trained model

```

## 🔧 Technologies & Libraries Used

### Core Stack
- **pandas**  - Data manipulation and analysis
- **numpy**  - Numerical computing

### Machine Learning
- **scikit-learn**  - Machine learning algorithms and utilities
- **joblib**  - Model serialization
- **statsmodels** - Statistical modeling

### Visualization
- **matplotlib** - Plotting library
- **seaborn** - Statistical data visualization

### Development & Deployment
- **jupyter** - Interactive notebooks
- **ipykernel** - Jupyter kernel
- **streamlit** - Web application framework

## 📊 Dataset & Methodology

**Dataset** 45,000 loan applications with 13 predictive features  
**Target**: Binary classification (Approved/Rejected)  
**Approach**: Supervised learning with tree-based ensemble methods

**Key Features**: Personal demographics, financial metrics, credit history, loan characteristics

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd 00_ClassificationModels-python
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## � Analysis & Findings

### Data Insights:
- **Class Imbalance**: 78% rejectd rate requiring Stratified strategie.
- **Feature Correlations**: Strong correlation between `person_age`, `person_emp_exp`, and `cb_person_cred_hist_length` indicates significant multicollinearity among these three variables.
- **Missing Data**: missing values handled by removing record. But no missing values.

### Model Evaluation Results:
| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| **Random Forest** | **0.98** | **92.9%** | **89.2%** | **77.5%** | **82.9%** |
| Gradient Boosting | 0.97 | 92.9% | 88.8% | 77.8% | 82.9% |


### Key Discoveries:
- **Top Predictors**: 
    - **Previous loan defaults** (combined importance: ~42%) : Consistent Across Models. Person's previous loan default history is one of the strongest indicators of future loan repayment behavior.
    - **loan interest rate** (13%) :  Higher interest rates often correlate with higher-risk loans, making this an important predictor.
    - **loan percent income** (~15%) : The ratio of loan amount to income directly relates to an applicant's ability to repay.
    
- **Model Stability**: CV standard deviation <0.02 across all metrics

## 🚀 Production Deployment

**Interactive Web Application** built with Streamlit for real-time loan approval predictions.

### Features:
- **Individual Predictions**: Web form interface for single applications
- **Batch Processing**: CSV upload for multiple loan assessments  
- **Probability Scoring**: Confidence levels and risk assessment
- **User-Friendly Interface**: Dropdown menus with data-driven options

## 🚀 How to use

There are two ways to try predictions:

### 🔹 Manual Entry
- Enter applicant loan details directly in the form.  
- Click **Predict** to see the result.  

### 🔹 CSV Upload
- Upload a CSV file containing one or more loan applications.  
- The app will output predictions for each row.  
- Example format:

```csv
person_age,person_gender,person_education,person_income,person_emp_exp,person_home_ownership,loan_amnt,loan_intent,loan_int_rate,loan_percent_income,cb_person_cred_hist_length,credit_score,previous_loan_defaults_on_file
25,male,graduate,40000,3,rent,12000,personal,12.5,0.30,5,700,no
45,female,not_graduate,80000,10,own,25000,home_improvement,10.0,0.20,15,720,yes

### Quick Start:
```bash
pip install -r requirements.txt
streamlit run LoanData_App_Model_Deployment.py
# Access at http://localhost:8501
```

**Business Value**: Enables instant loan pre-screening with 89% accuracy. 

## � Technical Implementation

### Model:
- **Algorithm**: Random Forest
- **Preprocessing**: One-hot encoding, balanced class weights
- **Validation**: 5-fold stratified cross-validation
- **Optimization**: GridSearchCV with 36 parameter combinations.

### Technical Skills Demonstrated:
- **Machine Learning**: Scikit-learn pipelines, hyperparameter tuning, model comparison
- **Visualization**: Matplotlib, Seaborn for EDA and results presentation
- **Deployment**: Streamlit web framework, model serialization with Joblib
- **Best Practices**: Cross-validation, stratified sampling, reproducible results

## 🎯 Project Outcomes

**Quantifiable Results:**
- **Model Accuracy**: 98% ROC-AUC score with robust cross-validation
documentation
- **Deployment Ready**: Production-quality web application with batch processing capabilities

**Skills Demonstrated:**
- End-to-end ML pipeline development
- Statistical analysis and feature engineering  
- Model comparison and hyperparameter optimization
- Web application development and deployment
- Professional documentation and code organization

---
