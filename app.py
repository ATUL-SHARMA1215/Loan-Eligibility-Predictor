import streamlit as st
import pickle
import numpy as np
import datetime
import base64
import matplotlib.pyplot as plt
import os
import platform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Smart TTS setup (disabled on Streamlit Cloud)
tts_available = False
engine = None
try:
    if platform.system() != "Linux" or not os.environ.get("HOME", "").startswith("/home/adminuser"):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        tts_available = True
except Exception:
    tts_available = False

# Load models
with open("model.pkl", "rb") as f:
    rf_model = pickle.load(f)
try:
    with open("logistic_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
except:
    lr_model = None

st.set_page_config(page_title="💸 Loan Eligibility Predictor", page_icon="💸", layout="centered")

st.markdown("""
    <style>
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .title-box {
        background-color: #4db6ac;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .result-box {
        border-radius: 10px;
        padding: 1rem;
        font-weight: bold;
        text-align: center;
        font-size: 18px;
        background-color: #f1f8e9;
        color: #33691e;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='title-box'>
        <h2>💸 Loan Eligibility Predictor</h2>
        <p>AI-powered predictor with model selection, charts, and voice feedback (local only).</p>
    </div>
""", unsafe_allow_html=True)

with st.form("loan_form"):
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
    col1, col2 = st.columns(2)
    with col1:
        Age = st.slider("Age", 18, 75, 30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Married", ["Yes", "No"])
        Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
        Credit_History = st.selectbox("Credit History", ["Good", "Bad"])
    with col2:
        ApplicantIncome = st.slider("Applicant Income", 0, 100000, 5000, step=1000)
        CoapplicantIncome = st.slider("Coapplicant Income", 0, 50000, 0, step=1000)
        LoanAmount = st.slider("Loan Amount (in thousands)", 0, 700, 150, step=10)
        Loan_Amount_Term = st.slider("Loan Term (in days)", 0, 480, 360, step=30)
        Property_Area = st.radio("Property Area", ["Urban", "Semiurban", "Rural"])

    show_graph = st.checkbox("📊 Show Insights (Income vs Loan)")
    submit = st.form_submit_button("🔍 Check Eligibility")

if submit:
    Gender_num = 1 if Gender == "Male" else 0
    Married_num = 1 if Married == "Yes" else 0
    Dependents_num = 3 if Dependents == "3+" else int(Dependents)
    Education_num = 0 if Education == "Graduate" else 1
    Self_Employed_num = 1 if Self_Employed == "Yes" else 0
    Credit_History_num = 1 if Credit_History == "Good" else 0
    Property_Area_num = {"Urban": 2, "Semiurban": 1, "Rural": 0}[Property_Area]

    features = np.array([[Age, Gender_num, Married_num, Dependents_num, Education_num, Self_Employed_num,
                          ApplicantIncome, CoapplicantIncome, LoanAmount,
                          Loan_Amount_Term, Credit_History_num, Property_Area_num]])

    if model_choice == "Logistic Regression" and lr_model:
        prediction = lr_model.predict(features)
        proba = lr_model.predict_proba(features)[0][1]
    else:
        prediction = rf_model.predict(features)
        proba = rf_model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        result_text = f"✅ Congratulations! You are eligible for the loan. Confidence: {proba * 100:.2f}%"
        st.markdown(f"<div class='result-box'>{result_text}</div>", unsafe_allow_html=True)
        if tts_available and engine:
            engine.say("You are eligible for the loan")
            engine.runAndWait()
    else:
        result_text = f"❌ Sorry, you're not eligible. Confidence: {(1 - proba) * 100:.2f}%"
        st.markdown(f"<div class='result-box' style='background-color:#ffebee; color:#b71c1c;'>{result_text}</div>", unsafe_allow_html=True)
        if tts_available and engine:
            engine.say("Sorry, you are not eligible for the loan")
            engine.runAndWait()

    if show_graph:
        st.subheader("📊 Data Insight Charts")
        fig, ax = plt.subplots()
        ax.bar(['Applicant', 'Coapplicant'], [ApplicantIncome, CoapplicantIncome], color=['#4db6ac', '#ff8a65'])
        ax.set_ylabel('Income (₹)')
        ax.set_title('Income Comparison')
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.bar(['Loan Requested'], [LoanAmount * 1000], color='#9575cd')
        ax2.set_ylabel('Loan Amount (₹)')
        ax2.set_title('Requested Loan Amount')
        st.pyplot(fig2)
