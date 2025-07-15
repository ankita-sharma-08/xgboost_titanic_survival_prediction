# Titanic Survival Prediction using XGBoost
for the above project app use the link to determine whether a passenger survived or not 

https://xgboost-prediction.streamlit.app/#email-ankitasharma7820-gmail-com

Here's a complete and clear **README** file for your Streamlit app that uses an XGBoost model (`xgboost.pkl`) to predict Titanic passenger survival:

---

# 🚢 Titanic Survival Prediction Web App

This is a **Streamlit web application** that predicts the survival of a passenger on the Titanic using a trained **XGBoost classification model**. The model is based on passenger details such as class, gender, age, fare, and more.

##  Project Structure

```
├── app.py                # Streamlit app script
├── xgboost.pkl           # Pre-trained XGBoost model file
```

---

##  Features

* Beautiful Titanic-themed UI
* Takes user input: Passenger Class, Sex, Age, Siblings/Spouses, Parents/Children, Fare, and Embarkation Port
* Encodes inputs and sends them through an XGBoost model
* Displays whether the passenger is likely to **survive** or **not survive**
* Includes author contact info

---

##  How to Run the App

### 1. Clone or Download the Project

```bash
git clone https://github.com/yourusername/titanic-survival-predictor.git
cd titanic-survival-predictor
```

### 2. Install Dependencies

Ensure Python 3.7+ is installed, then install the required libraries:

```bash
pip install streamlit pandas xgboost
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your web browser.

---

##  Model Details

* **Model Type**: XGBoost Classifier
* **Input Features**:

  * `pclass`: Passenger Class (1, 2, or 3)
  * `sex`: Gender (male/female)
  * `age`: Age of the passenger
  * `sibsp`: Number of siblings/spouses aboard
  * `parch`: Number of parents/children aboard
  * `fare`: Ticket fare
  * `embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

# Conclusion
This project demonstrates the end-to-end process of building a machine learning model for binary classification using the Titanic dataset. By leveraging XGBoost, the project showcases effective data handling, model training, and evaluation techniques, providing insights into the factors influencing survival on the Titanic. The saved model can be utilized for further analysis or deployment in applications requiring survival predictions.

---

## ⚠️ Disclaimer

This application is intended for **informational purposes only** and does not provide professional or life-critical predictions.
