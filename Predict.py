import joblib
import pandas as pd

#Load Saved Model
#===========================

model = joblib.load("loan_approval_model.pkl")
print("Model Loaded Successfully!")

#taking input from users
#=================================
gender = input("Enter Gender (Male/Female): ")
married = input("Married? (Yes/No): ")
dependents = input("Enter Dependents (0, 1, 2, 3+): ")
education = input("Education (Graduate/Not Graduate): ")
self_employed = input("Self Employed? (Yes/No): ")
income = float(input("Applicant Income: "))
co_income = float(input("Coapplicant Income: "))
loan_amount = float(input("Loan Amount: "))
credit_history = int(input("Credit History (1 or 0): "))

# Convert Input Data to dataframe
#=======================================
new_data = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": income,
    "Coapplicant Income": co_income,
    "LoanAmount": loan_amount,
    "Credit_History": credit_history
}])

#Convert Text to number
#========================
new_data = new_data.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

#predict new data
#===========================
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Predicted Result: Loan Approved")
else:
    print("Predicted Result: Loan Rejected")


#Predict New Data
#===========================

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Predicted Result: Loan Approved")
else:
    print("Predicted Result: Loan Rejected")













