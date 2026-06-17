import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page config
st.set_page_config(page_title="OLX Fraud Detection", layout="centered")

# Background function
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1570129477492-45c003edd2be");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        .block-container {
            background-color: rgba(0, 0, 0, 0.85);
            padding: 2rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
set_bg()

# Title
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>OLX Rental Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect suspicious rental listings using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")

# Input Section
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input(" Price (PKR)", min_value=0.0, step=1000.0)
    area  = st.number_input(" Area (sqft)", min_value=0.0, step=100.0)

with col2:
    bed  = st.number_input(" Bedrooms", min_value=0.0, step=1.0)
    bath = st.number_input(" Bathrooms", min_value=0.0, step=1.0)

st.markdown("---")

# Prediction
if st.button(" Predict"):
    if price == 0 or area == 0:
        st.warning(" Please enter valid property details")
    else:
        input_data = np.array([[price, area, bed, bath]])
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)[0]

        st.markdown("### Result")

        if prediction[0] == 1:
            st.error(f" Fraud Listing Detected (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f" Legitimate Listing (Confidence: {prob[0]*100:.2f}%)")

st.markdown("---")

# Footer
st.markdown(
    "<p style='text-align: center; font-size: 12px;'> Online Fraud Rental Detection</p>",
    unsafe_allow_html=True
)