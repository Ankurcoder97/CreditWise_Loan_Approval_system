import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("nb_model.pkl", "rb"))

st.set_page_config(page_title="CreditWise Loan System", layout="centered")
st.title("💳 CreditWise Loan Approval System")
st.markdown("Enter applicant details to check loan eligibility.")

# -------------------- USER INPUTS --------------------

Applicant_Income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
Coapplicant_Income = st.number_input("Co-applicant Income", min_value=0.0, value=2000.0)
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Dependents = st.number_input("Number of Dependents", min_value=0, value=1)
Existing_Loans = st.number_input("Existing Loans", min_value=0, value=0)
Savings = st.number_input("Savings Amount", min_value=0.0, value=10000.0)
Collateral_Value = st.number_input("Collateral Value", min_value=0.0, value=5000.0)
Loan_Amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0)
Loan_Term = st.number_input("Loan Term (months)", min_value=1, value=36)

Education_Level = st.selectbox("Education Level", [0, 1], 
                              format_func=lambda x: "Not Graduate" if x == 0 else "Graduate")

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

DTI_Ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=0.3, step=0.1, format="%.2f")
Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)

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

# -------------------- FEATURE VECTOR --------------------

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

# -------------------- PREDICTION --------------------
if st.button("Check Loan Eligibility", type="primary"):
    
    with st.spinner("Analyzing application..."):
        # Get prediction probabilities
        proba = model.predict_proba(features)
        
        # Assuming class 0 = Approved, class 1 = Rejected
        # Some models might have opposite ordering, let's check:
        approved_prob = proba[0][0]  # First probability
        rejected_prob = proba[0][1]  # Second probability
        
        # Optional: Check which class is which
        if hasattr(model, 'classes_'):
            st.write(f"Model classes: {model.classes_}")
            # If classes are ['Rejected', 'Approved'], adjust accordingly
            if len(model.classes_) == 2:
                if model.classes_[0] == 'Rejected' or model.classes_[0] == 1:
                    # Swap probabilities
                    approved_prob, rejected_prob = rejected_prob, approved_prob
        
        # Display probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Approval Probability", f"{approved_prob*100:.1f}%")
        with col2:
            st.metric("Rejection Probability", f"{rejected_prob*100:.1f}%")
        
        st.divider()
        
        # ---------- BUSINESS RULES VALIDATION ----------
        st.subheader("Business Rules Validation")
        rule_violations = []
        
        # Rule 1: Credit Score check
        if Credit_Score < 650:
            rule_violations.append("❌ Credit Score below 650")
        else:
            st.success("✅ Credit Score acceptable (≥ 650)")
        
        # Rule 2: Debt-to-Income Ratio check
        if DTI_Ratio > 0.5:
            rule_violations.append("❌ Debt-to-Income Ratio above 0.5")
        else:
            st.success("✅ Debt-to-Income Ratio acceptable (≤ 0.5)")
        
        # Rule 3: Existing Loans check
        if Existing_Loans >= 2:
            rule_violations.append("❌ Too many existing loans (≥ 2)")
        else:
            st.success("✅ Existing loans acceptable (< 2)")
        
        # Rule 4: Income to Loan Ratio (optional additional rule)
        total_income = Applicant_Income + Coapplicant_Income
        if total_income > 0 and Loan_Amount / total_income > 5:
            rule_violations.append("❌ Loan amount too high relative to income")
        elif total_income > 0:
            st.success(f"✅ Loan-to-Income ratio acceptable ({Loan_Amount/total_income:.2f})")
        
        st.divider()
        
        # ---------- FINAL DECISION ----------
        st.subheader("Final Decision")
        
        # Check if any business rules were violated
        if rule_violations:
            st.error("## ❌ LOAN REJECTED")
            st.write("**Reasons for rejection:**")
            for violation in rule_violations:
                st.write(f"- {violation}")
        else:
            # No rule violations, use model prediction with threshold
            if approved_prob >= 0.75:
                st.success("## ✅ LOAN APPROVED")
                st.balloons()
                st.write(f"**Approval confidence: {approved_prob*100:.1f}%**")
            else:
                st.error("## ❌ LOAN REJECTED")
                st.write(f"**Reason:** Model confidence too low ({approved_prob*100:.1f}% < 75% threshold)")
        
        # ---------- RECOMMENDATIONS ----------
        st.divider()
        st.subheader("💡 Recommendations")
        
        if rule_violations or approved_prob < 0.75:
            recommendations = []
            
            if Credit_Score < 650:
                recommendations.append("Improve your credit score to at least 650")
            if DTI_Ratio > 0.5:
                recommendations.append("Reduce your debt-to-income ratio below 0.5")
            if Existing_Loans >= 2:
                recommendations.append("Pay off some existing loans before applying")
            if Loan_Amount / (total_income + 1) > 5:
                recommendations.append("Consider requesting a smaller loan amount")
            if approved_prob < 0.75 and not rule_violations:
                recommendations.append("Increase your savings or collateral value")
                recommendations.append("Consider adding a co-applicant with higher income")
            
            if recommendations:
                st.write("To improve your chances:")
                for rec in recommendations:
                    st.write(f"• {rec}")
        else:
            st.write("• Your application meets all criteria!")
            st.write("• Consider proceeding with the loan documentation")

# -------------------- SIDEBAR INFO --------------------
with st.sidebar:
    st.header("ℹ️ Approval Criteria")
    st.write("**Hard Rules (Auto-Reject):**")
    st.write("- Credit Score < 650")
    st.write("- Debt-to-Income Ratio > 0.5")
    st.write("- Existing Loans ≥ 2")
    st.write("")
    st.write("**Model Threshold:**")
    st.write("- Approval probability ≥ 75%")
    st.write("")
    st.write("**Note:** All criteria must be met for approval.")
