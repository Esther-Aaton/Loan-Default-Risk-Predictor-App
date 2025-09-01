
import pandas as pd
import numpy as np
import streamlit as st
import joblib

#Loading the model
model = joblib.load("Loan_default_risk_predictor.pkl")

#Adding the streamlit title
st.title("Loan Default Risk Predictor App")

st.write("Fill in the Borrower's Financial and Demographic Information")

#User Input
referred = st.selectbox("Referral Status", ["True", "False"])
bank_account_type = st.selectbox("Bank Account Type", ["Other", "Savings", "Current"])
bank_name_clients = st.text_input("Enter your bank name", "GT Bank", max_chars=20)
employment_status_clients = st.selectbox("Employment Status", ["Permanent", "unknown", "Unemployed", "Self-Employed", "Student", "Retired", "Contract"])
age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-50", "51-65", "65+"])
payment_status = st.selectbox("Payment Status", ["on_time", "late"])
repayment_ratio = st.number_input("Repayment Ratio", 1.0, 1.5, 1.3)
loanamount = st.number_input("Loan Amount", 10000.0, 60000.0, 30000.0)
totaldue = st.number_input("Total Due", 10000.0, 70000.0, 35000.0)
termdays = st.slider("Term Days", 15.0, 90.0, 30.0)
avg_credit_score = st.number_input("Credit Score", 300.0, 850.0, 600.0)
loan_approval_time_min = st.slider("Loan Approval Time(min)", 60.0, 3000.0, 600.0)
loan_counts = st.number_input("Loan Counts", 1.0, 30.0, 10.0)
max_loan_amount = st.slider("Max Loan Collected", 10000.0, 60000.0, 30000.0)
avg_loan_amount = st.slider("Average Loan Collected", 10000.0, 60000.0, 30000.0)
longitude_gps = st.number_input("Enter Longitude (optional)", value=0.0, format="%.6f")
latitude_gps = st.number_input("Enter Latitude (optional)", value=0.0, format="%.6f")

#Prediction Button
if st.button("Predict the Loan Default Risk"):
  customer_info = {
    "Referral_status": [referred],
    "Bank_account_type": [bank_account_type],
    "Bank_name": [bank_name_clients],
    "Employment_status": [employment_status_clients],
    "Age_group": [age_group],
    "Payment_status": [payment_status],
    "Repayment_ratio":[repayment_ratio],
    "Loanamount":[loanamount],
    "Totaldue":[totaldue],
    "Termdays":[termdays],
    "Avg_credit_score":[avg_credit_score],
    "Loan_approval_time_min":[loan_approval_time_min],
    "Loan_counts":[loan_counts],
    "Max_loan_amount":[max_loan_amount],
    "Avg_loan_amount":[avg_loan_amount],
    "Longitude_gps":[longitude_gps],
    "Latitude_gps":[latitude_gps]
  }
  #Converting into a DataFrame
  input_df = pd.DataFrame(customer_info)

  #predicting my model
  try:
    loan_default = model.predict(input_df)[0]
    prediction = np.expm1(loan_default)
    st.success(f"Borrower's Loan Default Risk is:{prediction:.2f}")
  except Exception as e:
    st.error(f"Prediction error: {e}")
