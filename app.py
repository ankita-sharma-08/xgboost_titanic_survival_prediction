import streamlit as st
import pickle
import numpy as np

# Load the model
with open('xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app title
st.title("XGBoost Model Prediction App")

# Set background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://your-image-url.com/background.jpg');
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Personal Details Section
st.sidebar.header("Personal Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
email = st.sidebar.text_input("Email")

# Input features
st.header("Input Features")
feature1 = st.number_input("Feature 1", min_value=0.0, max_value=100.0, value=50.0)
feature2 = st.number_input("Feature 2", min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.number_input("Feature 3", min_value=0.0, max_value=100.0, value=50.0)

# Create a button for prediction
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([[feature1, feature2, feature3]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.success(f"The predicted value is: {prediction[0]}")

# Display personal details
if st.sidebar.button("Submit"):
    st.sidebar.success(f"Details Submitted: {name}, Age: {age}, Email: {email}")
