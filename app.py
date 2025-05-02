# Import necessary libraries
import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Load the saved model
with open("xgboost.pkl", "rb") as f:
    model = pickle.load(f)

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
        st.success("The passenger is predicted to survive.")
    else:
        st.error("The passenger is predicted not to survive.")
