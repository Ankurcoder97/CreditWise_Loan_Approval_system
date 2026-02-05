import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("nb_model.pkl", "rb"))

st.set_page_config(page_title="CreditWise Loan System", layout="centered")
st.title("💳 CreditWise Loan Approval System")

st.markdown("Enter applicant details to check loan eligibility.")

# -------------------- USER INPUTS --------------------

Applicant_Income = st.number_input("Applicant Income", min_value=0.0)
Coapplicant_Income = st.number_input("Co-applicant Income", min_value=0.0)
Age = st.number_input("Age", min_value=18, max_value=100)
Dependents = st.number_input("Number of Dependents", min_value=0)
Existing_Loans = st.number_input("Existing Loans", min_value=0)
Savings = st.number_input("Savings Amount", min_value=0.0)
Collateral_Value = st.number_input("Collateral Value", min_value=0.0)
Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
Loan_Term = st.number_input("Loan Term (months)", min_value=1)

Education_Level = st.selectbox("Education Level", [0, 1])  
# 0 = Not Graduate, 1 = Graduate (must match notebook)

employment = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Unemployed"]
)

marital = st.selectbox("Marital Status", ["Single", "Married"])

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Car", "Education", "Home", "Personal"]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

employer_category = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)

DTI_Ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0)
Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900)

# -------------------- ONE-HOT ENCODING --------------------

Employment_Status_Salaried = 1 if employment == "Salaried" else 0
Employment_Status_Self_employed = 1 if employment == "Self-employed" else 0
Employment_Status_Unemployed = 1 if employment == "Unemployed" else 0

Marital_Status_Single = 1 if marital == "Single" else 0

Loan_Purpose_Car = 1 if loan_purpose == "Car" else 0
Loan_Purpose_Education = 1 if loan_purpose == "Education" else 0
Loan_Purpose_Home = 1 if loan_purpose == "Home" else 0
Loan_Purpose_Personal = 1 if loan_purpose == "Personal" else 0

Property_Area_Semiurban = 1 if property_area == "Semiurban" else 0
Property_Area_Urban = 1 if property_area == "Urban" else 0

Gender_Male = 1 if gender == "Male" else 0

Employer_Category_Government = 1 if employer_category == "Government" else 0
Employer_Category_MNC = 1 if employer_category == "MNC" else 0
Employer_Category_Private = 1 if employer_category == "Private" else 0
Employer_Category_Unemployed = 1 if employer_category == "Unemployed" else 0

# -------------------- FEATURE ENGINEERING --------------------

DTI_Ratio_sq = DTI_Ratio ** 2
Credit_Score_sq = Credit_Score ** 2

# -------------------- FEATURE VECTOR (ORDER MATTERS!) --------------------

features = np.array([[
    Applicant_Income,
    Coapplicant_Income,
    Age,
    Dependents,
    Existing_Loans,
    Savings,
    Collateral_Value,
    Loan_Amount,
    Loan_Term,
    Education_Level,
    Employment_Status_Salaried,
    Employment_Status_Self_employed,
    Employment_Status_Unemployed,
    Marital_Status_Single,
    Loan_Purpose_Car,
    Loan_Purpose_Education,
    Loan_Purpose_Home,
    Loan_Purpose_Personal,
    Property_Area_Semiurban,
    Property_Area_Urban,
    Gender_Male,
    Employer_Category_Government,
    Employer_Category_MNC,
    Employer_Category_Private,
    Employer_Category_Unemployed,
    DTI_Ratio_sq,
    Credit_Score_sq
]])

assert features.shape[1] == 27

# -------------------- PREDICTION --------------------

if st.button("Check Loan Eligibility"):
    prediction = model.predict(features)
    proba = model.predict_proba(features)

    st.write("Raw prediction:", prediction)
    st.write("Probabilities [Approved, Rejected]:", proba)

    if prediction[0] == 0:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
