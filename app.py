import os
import pickle
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="⚡",
    layout="centered"
)

# Custom CSS for gorgeous styling and layout spacing
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        color: #f8fafc;
        background-attachment: fixed;
    }

    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding-top: 1rem;
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(to right, #c084fc, #f472b6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* Container Box Styling */
    div.block-container {
        padding: 2rem;
        max-width: 850px;
    }

    /* Form Labels */
    label {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        margin-bottom: 0.3rem !important;
    }

    /* Input fields and selectboxes */
    .stSelectbox div[data-baseweb="select"], .stNumberInput input {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
    }

    /* Submit Button Styling */
    div.stButton > button {
        width: 100%;
        margin-top: 1.5rem;
        padding: 0.85rem;
        background: linear-gradient(135deg, #a855f7, #ec4899);
        border: none;
        border-radius: 12px;
        color: white;
        font-size: 1.05rem;
        font-weight: 700;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 10px 25px -5px rgba(236, 72, 153, 0.4);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px -5px rgba(236, 72, 153, 0.6);
        border: none;
        color: white;
    }

    /* Result Cards */
    .result-card-success {
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        animation: fadeIn 0.4s ease-in-out;
    }

    .result-card-danger {
        margin-top: 2rem;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        animation: fadeIn 0.4s ease-in-out;
    }

    .result-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }

    .result-value-success {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4ade80;
    }

    .result-value-danger {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f87171;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Load Model safely
MODEL_PATH = "naiveB.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Header section
st.markdown("""
<div class="main-header">
    <h1>Employee Retention Predictor</h1>
    <p>Powered by Naive Bayes Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Error: Model file 'naiveB.pkl' not found in your repository root directory.")
else:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            education = st.selectbox(
                "Education Level", 
                options=[("Bachelors", 0), ("Masters", 1), ("PHD", 2)], 
                format_func=lambda x: x[0]
            )[1]
            joining_year = st.number_input("Joining Year", min_value=2010, max_value=2030, value=2018, step=1)
            city = st.selectbox(
                "City", 
                options=[("Bangalore", 0), ("Pune", 1), ("New Delhi", 2)], 
                format_func=lambda x: x[0]
            )[1]
            payment_tier = st.selectbox(
                "Payment Tier", 
                options=[("Tier 1 (High)", 1), ("Tier 2 (Medium)", 2), ("Tier 3 (Standard)", 3)], 
                format_func=lambda x: x[0]
            )[1]

        with col2:
            age = st.number_input("Age", min_value=18, max_value=70, value=28, step=1)
            gender = st.selectbox(
                "Gender", 
                options=[("Male", 0), ("Female", 1)], 
                format_func=lambda x: x[0]
            )[1]
            ever_benched = st.selectbox(
                "Ever Benched", 
                options=[("No", 0), ("Yes", 1)], 
                format_func=lambda x: x[0]
            )[1]
            experience = st.number_input("Experience in Current Domain (Years)", min_value=0, max_value=20, value=3, step=1)

        submit_button = st.form_submit_button("Predict Retention Status")

    if submit_button:
        try:
            features = np.array([[education, joining_year, city, payment_tier, age, gender, ever_benched, experience]])
            prediction = model.predict(features)[0]

            if prediction == 1:
                st.markdown("""
                <div class="result-card-danger">
                    <div class="result-title">Prediction Result</div>
                    <div class="result-value-danger">⚠️ Employee Likely to Leave (High Churn Risk)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card-success">
                    <div class="result-title">Prediction Result</div>
                    <div class="result-value-success">✨ Employee Likely to Stay (Retained)</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
