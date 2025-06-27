import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("loan_data.csv")

# Ensure 'Age' exists
if 'Age' not in df.columns:
    df['Age'] = np.random.randint(21, 65, size=len(df))

# Drop Loan_ID if present
df.drop(columns=['Loan_ID'], inplace=True, errors='ignore')

# Forward fill missing values
# Fill missing values
df.fillna(method='ffill', inplace=True)  # Forward fill
df.fillna(method='bfill', inplace=True)  # Backward fill (in case ffill fails)
df.dropna(inplace=True)                  # Final drop for stubborn NaNs

# Confirm no missing values remain
print("✅ Missing values after cleanup:")
print(df.isnull().sum().sum(), "NaNs remaining")

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col != 'Loan_Status':
        df[col] = le.fit_transform(df[col])

# Encode target variable
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Feature list
features = [
    'Age', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

X = df[features]
y = df['Loan_Status']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Train and Save Random Forest Model ----
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
with open("model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# ---- Train and Save Logistic Regression Model ----
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# ---- Evaluation ----
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print(f"🎯 Random Forest Accuracy: {rf_model.score(X_test, y_test):.2f}")
print(f"🎯 Logistic Regression Accuracy: {lr_model.score(X_test, y_test):.2f}")

# ---- ROC Curve ----
rf_probs = rf_model.predict_proba(X_test)[:, 1]
lr_probs = lr_model.predict_proba(X_test)[:, 1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label='Random Forest')
plt.plot(lr_fpr, lr_tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# ---- Confusion Matrices ----
ConfusionMatrixDisplay(
    confusion_matrix(y_test, rf_pred),
    display_labels=["Not Eligible", "Eligible"]
).plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.close()

ConfusionMatrixDisplay(
    confusion_matrix(y_test, lr_pred),
    display_labels=["Not Eligible", "Eligible"]
).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("lr_confusion_matrix.png")
plt.close()

print("✅ Models trained and evaluation charts saved: ")
print("   - roc_curve.png")
print("   - rf_confusion_matrix.png")
print("   - lr_confusion_matrix.png")