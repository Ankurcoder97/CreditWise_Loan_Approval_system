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
        
        # Get actual prediction
        prediction = model.predict(features)[0]
        
        # Model classes are [0 1] where 0=Approved, 1=Rejected
        approved_prob = proba[0][0]  # Probability of class 0 (Approved)
        rejected_prob = proba[0][1]  # Probability of class 1 (Rejected)
        
        # ---------- BUSINESS RULES VALIDATION ----------
        st.subheader("📋 Business Rules Validation")
        
        rule_violations = []
        warnings = []
        
        # Rule 1: Credit Score check (HARD RULE)
        if Credit_Score < 650:
            rule_violations.append(f"Credit Score below 650 (Current: {Credit_Score})")
        else:
            st.success(f"✅ Credit Score: {Credit_Score} (≥ 650)")
        
        # Rule 2: Debt-to-Income Ratio check (HARD RULE)
        if DTI_Ratio > 0.5:
            rule_violations.append(f"Debt-to-Income Ratio above 0.5 (Current: {DTI_Ratio:.2f})")
        else:
            st.success(f"✅ Debt-to-Income Ratio: {DTI_Ratio:.2f} (≤ 0.5)")
        
        # Rule 3: Existing Loans check (HARD RULE)
        if Existing_Loans >= 2:
            rule_violations.append(f"Too many existing loans: {Existing_Loans} (Maximum: 1)")
        else:
            st.success(f"✅ Existing Loans: {Existing_Loans} (≤ 1)")
        
        # Rule 4: Income to Loan Ratio (WARNING, not hard rule)
        total_income = Applicant_Income + Coapplicant_Income
        loan_to_income_ratio = Loan_Amount / total_income if total_income > 0 else 0
        
        if loan_to_income_ratio > 5:
            warnings.append(f"High loan-to-income ratio: {loan_to_income_ratio:.2f} (Recommended: ≤ 5)")
            st.warning(f"⚠️ Loan-to-Income Ratio: {loan_to_income_ratio:.2f} (High)")
        elif total_income > 0:
            st.success(f"✅ Loan-to-Income Ratio: {loan_to_income_ratio:.2f} (≤ 5)")
        
        # Rule 5: Model Confidence (WARNING if low)
        if approved_prob < 0.75:
            warnings.append(f"Low model confidence: {approved_prob*100:.1f}% (Threshold: 75%)")
        
        st.divider()
        
        # ---------- MODEL PREDICTION RESULTS ----------
        st.subheader("🤖 Model Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            # Show color-coded probability
            if approved_prob >= 0.75:
                st.metric("Approval Probability", f"{approved_prob*100:.1f}%", 
                         delta="High Confidence", delta_color="normal")
            elif approved_prob >= 0.5:
                st.metric("Approval Probability", f"{approved_prob*100:.1f}%", 
                         delta="Medium Confidence", delta_color="off")
            else:
                st.metric("Approval Probability", f"{approved_prob*100:.1f}%", 
                         delta="Low Confidence", delta_color="inverse")
        
        with col2:
            # Show what the model would predict
            model_decision = "APPROVED" if prediction == 0 else "REJECTED"
            if prediction == 0:
                st.success(f"Model Decision: {model_decision}")
            else:
                st.error(f"Model Decision: {model_decision}")
        
        # Explain the model's thinking
        if approved_prob > 0.9:
            st.info("The model is very confident in its prediction.")
        elif approved_prob > 0.7:
            st.info("The model is fairly confident in its prediction.")
        else:
            st.info("The model has low confidence in its prediction.")
        
        st.divider()
        
        # ---------- FINAL DECISION ----------
        st.subheader("⚖️ Final Decision")
        
        # Check if any HARD business rules were violated
        if rule_violations:
            st.error("## ❌ LOAN REJECTED")
            st.write("**Reasons for automatic rejection (business rules):**")
            for violation in rule_violations:
                st.write(f"• {violation}")
            
            # Show warnings if any
            if warnings:
                st.write("**Additional concerns:**")
                for warning in warnings:
                    st.write(f"• ⚠️ {warning}")
                
            st.write("\n**Note:** Business rules take precedence over model predictions for risk management.")
            
        else:
            # No hard rule violations, now check model confidence
            if approved_prob >= 0.75:
                st.success("## ✅ LOAN APPROVED")
                st.balloons()
                st.write(f"**All criteria met with {approved_prob*100:.1f}% confidence**")
                
                # Show any warnings as notes
                if warnings:
                    st.write("**Notes:**")
                    for warning in warnings:
                        st.write(f"• {warning}")
            else:
                st.error("## ❌ LOAN REJECTED")
                st.write(f"**Reason:** Model confidence too low ({approved_prob*100:.1f}% < 75% threshold)")
                
                # If model actually predicted approved but confidence is low
                if prediction == 0:
                    st.write("Although the model predicts approval, confidence is below the required threshold.")
        
        # ---------- RECOMMENDATIONS ----------
        st.divider()
        st.subheader("💡 Recommendations")
        
        if rule_violations:
            st.write("**To fix automatic rejection issues:**")
            for violation in rule_violations:
                if "Credit Score" in violation:
                    st.write("• Improve credit score to at least 650")
                    st.write("  - Pay bills on time")
                    st.write("  - Reduce credit card utilization")
                    st.write("  - Check credit report for errors")
                elif "Debt-to-Income" in violation:
                    st.write("• Reduce debt-to-income ratio below 0.5")
                    st.write("  - Pay down existing debt")
                    st.write("  - Increase your income")
                elif "existing loans" in violation:
                    st.write("• Reduce number of existing loans")
                    st.write("  - Consolidate loans if possible")
                    st.write("  - Pay off smaller loans first")
        
        elif approved_prob < 0.75:
            st.write("**To improve model confidence:**")
            if loan_to_income_ratio > 3:
                st.write("• Request a smaller loan amount")
            if Savings < Loan_Amount * 0.2:
                st.write("• Increase your savings")
            if Collateral_Value < Loan_Amount * 0.5:
                st.write("• Provide additional collateral")
            st.write("• Consider adding a co-applicant with stable income")
            st.write("• Choose a shorter loan term")
        
        else:
            # Approved
            st.success("**Your application looks strong!**")
            st.write("• Proceed with loan documentation")
            st.write("• Keep maintaining good financial habits")
        
        # ---------- SUMMARY TABLE ----------
        st.divider()
        st.subheader("📊 Application Summary")
        
        summary_data = {
            "Criteria": ["Credit Score", "DTI Ratio", "Existing Loans", "Model Confidence", "Total Income", "Loan Amount", "Loan-to-Income"],
            "Value": [
                f"{Credit_Score}",
                f"{DTI_Ratio:.2f}",
                f"{Existing_Loans}",
                f"{approved_prob*100:.1f}%",
                f"${total_income:,.0f}",
                f"${Loan_Amount:,.0f}",
                f"{loan_to_income_ratio:.2f}"
            ],
            "Status": [
                "✅ Pass" if Credit_Score >= 650 else "❌ Fail",
                "✅ Pass" if DTI_Ratio <= 0.5 else "❌ Fail",
                "✅ Pass" if Existing_Loans < 2 else "❌ Fail",
                "✅ Pass" if approved_prob >= 0.75 else "❌ Fail",
                "-",
                "-",
                "⚠️ High" if loan_to_income_ratio > 5 else "✅ Good"
            ]
        }
        
        # Display as table
        import pandas as pd
        df_summary = pd.DataFrame(summary_data)
        st.table(df_summary)

# -------------------- SIDEBAR INFO --------------------
with st.sidebar:
    st.header("ℹ️ Approval Process")
    st.write("**Two-Step Verification:**")
    st.write("1. **Business Rules (Auto-Reject)**")
    st.write("   - Credit Score ≥ 650")
    st.write("   - DTI Ratio ≤ 0.5")
    st.write("   - Existing Loans < 2")
    st.write("")
    st.write("2. **Model Confidence**")
    st.write("   - Approval probability ≥ 75%")
    st.write("")
    st.write("**Both conditions must be met for approval.**")
    st.write("")
    st.write("**Note:** The AI model prediction (shown above) is overridden by business rules for risk control.")
