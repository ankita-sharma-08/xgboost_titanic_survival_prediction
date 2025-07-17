# Import necessary libraries
import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Load the saved model
with open("xgboost.pkl", "rb") as f:
    model = pickle.load(f)
    
# Set background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://cdn.pixabay.com/photo/2022/01/27/17/51/titanic-6972725_1280.jpg');
        background-size: cover;
        background-position: center;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Function to make predictions
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Prepare the input data
    input_data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'embarked': [embarked]
    })
    
    # Convert categorical variables
    input_data['sex'] = input_data['sex'].map({'male': 0, 'female': 1})
    input_data['embarked'] = input_data['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    
    # Create DMatrix
    dinput = xgb.DMatrix(input_data, enable_categorical=True)
    
    # Make prediction
    prediction = model.predict(dinput)
    return 1 if prediction[0] > 0.5 else 0

# Streamlit app layout
st.title("Titanic Survival Prediction")

# User input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked = st.selectbox("Embarked", ['C', 'Q', 'S'])

# Prediction button
if st.button("Predict"):
    result = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
    if result == 1:
        st.markdown("<h3 style='color: green;'> 🟢 The passenger is predicted to survive !!.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: red;'> 🔴 The passenger is predicted not to survive !!.</h3>", unsafe_allow_html=True)



st.markdown("------------") # Horizontal line 
st.markdown("<div style='text-align: right;'><h4 style='color: white;'>Developed by: Ankita Sharma</h4>", unsafe_allow_html=True) 
st.markdown("<div style='text-align: right;'><h4 style='color: white;'>Email: ankitasharma7820@gmail.com</h4>", unsafe_allow_html=True)

# # Disclaimer message
st.markdown("------------")
disclaimer = "This prediction is based on a machine learning model and may not be accurate."
       
       return jsonify({'survived': y_pred[0], 'disclaimer': disclaimer})

            
