import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import generic
from pandas.core.interchange import column
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib

#===================================
# Load the loan dataset
#==================================
df=pd.read_csv("loan_approval_dataset.csv")


# Display the first five records
#================================
print(df.head(5))

# Check dataset size
#==============================
print("\n DataSet Information:")
print(df.shape)

#=================================
###############Check Column Names
#=================================

print("\n Columns: Names:")
print(df.columns.to_list())


# Check column names and data types
#======================================
print("\n Dataset Information")
print(df.info())


#Numerical summary
#======================================
print("\n Statistical Summary")
print(df.describe())

# Check missing values
#=========================
print("\nMissing Values")
print(df.isnull().sum())

#=================================
# Check Duplicate Rows
#=================================
print("\n Duplicate Rows")
print(df.duplicated().sum())

#=================================
# Loan Status Analysis
#=================================
print("\n Loan Status Count")
print(df["Loan_Status"].value_counts())

#=================================
# Loan Status Distribution
#=================================

sns.countplot(data=df, x="Loan_Status")

plt.title("Loan Approval Status")
plt.xlabel("Loan Status (Y = Approved, N = Rejected)")
plt.ylabel("Number of Applications")

plt.show()

#=================================
# Gender vs Loan Status
#=================================

sns.countplot(data=df, x="Gender", hue="Loan_Status")

plt.title("Gender vs Loan Status")
plt.xlabel("Gender")
plt.ylabel("Number of Applications")

plt.show()

#========================================
# Convert Categorical Data into Numbers
#========================================

X=df.drop(columns=["Loan_Status"])
X=pd.get_dummies(X,drop_first=True,dtype=int)
print("\n Encoded Feature:")
print(X.head())

print("\n Feature Columns Names:")
print(X.columns.to_list())

#========================================
# Separate Target
#========================================

y=df["Loan_Status"].map({"N":0,"Y":1})
print("\nTarget Values:")
print(y)

print("\nX Shape:",X.shape)
print("\n Y Shape:",y.shape)

#=================================
# Train-Test Split
#=================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining and Testing Data:")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#=================================
# Create Model
#=================================

model=LogisticRegression(max_iter=1000)

#=================================
# Train the Model
#=================================
model.fit(X_train, y_train)
print("Model Trained")


# Predict the Model
#=================================

y_pred=model.predict(X_test)
print("Model Predicted")
print(y_pred)


# Compare Actual and Predicted Results
#=================================

print("Actual Result",y_test.tolist())
print("Predicted Result",y_pred.tolist())


# Model Evaluation
#=================================
accuracy=accuracy_score(y_test,y_pred)
print(" Model Accuracy:",accuracy* 100,"%")

# Confusion Matrix
#=================================
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)


#=================================
# Classification Report
#=================================

print("\n Classification Report:")
print(classification_report(y_test,y_pred,zero_division=0))

joblib.dump(model,"loan_approval_model.pkl")
print("Model Saved Successfully")







