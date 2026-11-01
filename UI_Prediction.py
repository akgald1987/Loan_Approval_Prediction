
#=================================
# Loan Approval Prediction UI
#=================================

import streamlit as st
import pandas as pd
import joblib

#=================================
# Load Saved Model
#=================================

model = joblib.load("loan_approval_model.pkl")

#=================================
# Create UI Heading
#=================================

st.title("Loan Approval Prediction System")
st.write("Enter applicant details to predict loan status.")

#=================================
# Take Input from User
#=================================

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.number_input(
    "Dependents", min_value=0, max_value=3, step=1
)
education = st.selectbox(
    "Education", ["Graduate", "Not Graduate"]
)
self_employed = st.selectbox(
    "Self Employed", ["Yes", "No"]
)

income = st.number_input("Applicant Income", min_value=0)
co_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])

#=================================
# Predict Loan Status
#=================================

if st.button("Predict Loan Status"):

    new_data = pd.DataFrame([{
        "Dependents": dependents,
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan_amount,
        "Credit_History": credit_history,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Married_Yes": 1 if married == "Yes" else 0,
        "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
        "Self_Employed_Yes": 1 if self_employed == "Yes" else 0
    }])

    #=================================
    # Match Model Feature Columns
    #=================================

    new_data = new_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    #=================================
    # Display Prediction
    #=================================

    prediction = model.predict(new_data)

    if prediction[0] == 1:
        st.success("Prediction: Loan Approved")
    else:
        st.error("Prediction: Loan Rejected")
