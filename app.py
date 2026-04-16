import streamlit as st
import pickle
import numpy as np

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="ML Regression App",
    page_icon="✨",
    layout="centered"
)

# -----------------------
# CUSTOM CSS (FANCY UI)
# -----------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
    }

    .main {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
    }

    h1 {
        text-align: center;
        color: #1f4037;
    }

    .stButton>button {
        background: linear-gradient(90deg, #1f4037, #99f2c8);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        border: none;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #99f2c8, #1f4037);
        color: black;
    }

    .result-box {
        background: #1f4037;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    with open("Regression.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -----------------------
# TITLE
# -----------------------
st.markdown("<h1>✨ Regression Predictor</h1>", unsafe_allow_html=True)
st.write("Enter values below to get predictions")

# -----------------------
# INPUT FIELDS
# ⚠️ EDIT THESE BASED ON YOUR DATASET
# -----------------------

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)

# -----------------------
# PREDICTION
# -----------------------
if st.button("🚀 Predict"):
    try:
        input_data = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(input_data)

        st.markdown(
            f'<div class="result-box">Prediction: {prediction[0]:.2f}</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
