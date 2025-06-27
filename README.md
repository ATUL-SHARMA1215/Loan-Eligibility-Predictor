# 💸 Loan Eligibility Predictor

A smart and interactive AI-powered web application built with **Streamlit** that predicts whether a user is eligible for a loan based on financial and personal details.

![Demo](https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg)

---

## 🚀 Features

✅ Real-time loan eligibility prediction  
✅ Confidence/probability score from ML model  
✅ Downloadable result in `.txt` and `.pdf` formats  
✅ Visual insights: income vs loan graphs  
✅ Mobile-friendly, clean responsive UI  
✅ Voice feedback (locally with `pyttsx3`)  
✅ Dark mode (optional)  

---

## 🧠 Machine Learning

Model: **Random Forest Classifier**  
Trained using the popular [Loan Prediction dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) from Kaggle.

### Input Features:
- Gender
- Marital Status
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/loan-eligibility-predictor.git
cd loan-eligibility-predictor
pip install -r requirements.txt
streamlit run app.py