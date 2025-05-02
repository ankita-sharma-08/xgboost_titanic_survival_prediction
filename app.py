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
        background-image: url('https://bs-uploads.toptal.io/blackfish-uploads/components/blog_post_page/4086440/cover_image/retina_1708x683/cover-color-psychology-03b7cbed78d0c396891bd10a9b1b855f.png');
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Input features
st.header("Enter the Details")
feature1 = st.number_input("Feature 1", min_value=0.0, max_value=100.0, value=50.0)
feature2 = st.number_input("Feature 2", min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.number_input("Feature 3", min_value=0.0, max_value=100.0, value=50.0)

# Create a button for prediction
if st.button("Predict"):
    # Prepare the input data for prediction
    data = np.array([[feature1, feature2, feature3]])
    
    # Make prediction
    prediction = model.predict(data)
    
    # Display the prediction
    st.success(f"The predicted value is: {prediction[0]}")




# Footer with name and email - shows on bottom of the app
footer_html = f"""
    <style>
        .footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(0,0,0,0.5);
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            z-index: 9999;
        }}
    </style>
    <div class="footer">
        Developed by {name if name else 'Ankita Sharma'} | Email: {email if email else 'ankitasharma7820@gmail.com'}
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
