import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("loan_data.csv")

# Add 'Age' column manually if not in dataset (demo purpose)
if 'Age' not in df.columns:
    df['Age'] = np.random.randint(21, 65, size=len(df))

# Drop Loan_ID if exists
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col != 'Loan_Status':
        df[col] = le.fit_transform(df[col])

df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Final feature list (must match app.py)
features = [
    'Age', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

X = df[features]
y = df['Loan_Status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
with open("model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# Evaluation
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print("🎯 Random Forest Accuracy:", rf_model.score(X_test, y_test))
print("🎯 Logistic Regression Accuracy:", lr_model.score(X_test, y_test))

# ROC Curves
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

# Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, rf_pred), display_labels=["Not Eligible", "Eligible"]).plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.close()

ConfusionMatrixDisplay(confusion_matrix(y_test, lr_pred), display_labels=["Not Eligible", "Eligible"]).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("lr_confusion_matrix.png")
plt.close()

print("✅ Models trained and evaluation charts saved (roc_curve.png, rf_confusion_matrix.png, lr_confusion_matrix.png)")