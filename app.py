import streamlit as st
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM CSS FOR FANCY UI ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* Typography and Header styling */
    h1 {
        color: #1E3A8A;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Customizing the prediction button */
    .stButton>button {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: #ffffff;
    }
    
    /* Card layout for the result */
    .result-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 30px;
        border-top: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("<h1>💼 Career Value Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Salary Estimation based on Experience</div>", unsafe_allow_html=True)
st.write("---")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    with open('Regression.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file 'Regression.pkl' not found. Please upload it to your repository.")
    st.stop()

# --- INPUT SECTION ---
st.markdown("### ✨ Enter Your Details")
# Using a slider for a smooth user experience
years_experience = st.slider(
    "Years of Experience", 
    min_value=0.0, 
    max_value=50.0, 
    value=5.0, 
    step=0.5,
    help="Slide to select your total years of professional experience."
)

st.write("") # Spacer

# --- PREDICTION SECTION ---
if st.button("🔮 Predict Salary"):
    with st.spinner("Analyzing market data..."):
        # The model expects a 2D array for features
        input_data = np.array([[years_experience]])
        prediction = model.predict(input_data)[0]
        
        # Trigger success animation
        st.balloons()
        
        # Display the result in the custom styled card
        st.markdown(f"""
            <div class="result-card">
                <p style='color: #64748B; font-size: 1.2rem; margin-bottom: 5px;'>Estimated Compensation</p>
                <h2 style='color: #10B981; font-size: 3rem; margin: 0;'>${prediction:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)
