
import streamlit as st
import pickle
import numpy as np
import datetime
import base64
import matplotlib.pyplot as plt
import platform
import os

# 🔄 Smart TTS initialization: only for local use
tts_available = False
if platform.system() != "Linux" or os.environ.get("HOME") != "/home/adminuser":
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        tts_available = True
    except:
        tts_available = False

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="💸 Loan Eligibility Predictor", page_icon="💸", layout="centered")



# CSS Styling
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

# Title
st.markdown("""
    <div class='title-box'>
        <h2>💸 Loan Eligibility Predictor</h2>
        <p>AI-powered, mobile-ready and interactive predictor with TTS feature.</p>
    </div>
""", unsafe_allow_html=True)

# Input form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
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

# Prediction logic
if submit:
    Gender = 1 if Gender == "Male" else 0
    Married = 1 if Married == "Yes" else 0
    Dependents = 3 if Dependents == "3+" else int(Dependents)
    Education = 0 if Education == "Graduate" else 1
    Self_Employed = 1 if Self_Employed == "Yes" else 0
    Credit_History = 1 if Credit_History == "Good" else 0
    Property_Area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[Property_Area]

    features = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                          ApplicantIncome, CoapplicantIncome, LoanAmount,
                          Loan_Amount_Term, Credit_History, Property_Area]])

    prediction = model.predict(features)
    proba = model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        result_text = f"✅ Congratulations! You are eligible for the loan. Confidence: {proba * 100:.2f}%"
        st.markdown(f"<div class='result-box'>{result_text}</div>", unsafe_allow_html=True)
        engine.say("You are eligible for the loan")
        engine.runAndWait()
    else:
        result_text = f"❌ Sorry, you're not eligible. Confidence: {(1 - proba) * 100:.2f}%"
        st.markdown(f"<div class='result-box' style='background-color:#ffebee; color:#b71c1c;'>{result_text}</div>", unsafe_allow_html=True)
        if tts_available:
            engine.say("You are eligible for the loan")
            engine.runAndWait()
            if tts_available:
                engine.say("Sorry, you're not eligible")
                engine.runAndWait()
    # Download report
    report = f"Loan Eligibility Report\nDate: {datetime.datetime.now()}\n\n{result_text}\n\nDetails:\nGender: {Gender}\nMarried: {Married}\nDependents: {Dependents}\nEducation: {Education}\nIncome: ₹{ApplicantIncome}\nCoapplicant: ₹{CoapplicantIncome}\nLoan: ₹{LoanAmount * 1000}\nProperty Area: {Property_Area}"
    st.download_button("📄 Download Result", data=report, file_name="loan_eligibility.txt")

    # Optional Graph
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
