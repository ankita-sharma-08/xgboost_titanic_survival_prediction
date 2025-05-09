# Titanic Survival Prediction using XGBoost
for the above project app use the link to determine whether a passenger survived or not 

https://xgboost-prediction.streamlit.app/#email-ankitasharma7820-gmail-com

# Overview
This project aims to predict the survival of passengers on the Titanic using machine learning techniques, specifically the XGBoost algorithm. The Titanic dataset is a well-known dataset used for binary classification tasks, where the goal is to determine whether a passenger survived or not based on various features such as age, gender, class, and fare.

Objectives
1. Data Retrieval: Connect to a MySQL database to retrieve the Titanic dataset.

2. Data Preprocessing: Clean and preprocess the data to handle missing values and encode categorical variables.

3. Model Training: Train an XGBoost model to predict survival based on selected features.

4. Model Evaluation: Evaluate the model's performance using accuracy and classification metrics.

5. Model Persistence: Save the trained model for future use.

# Steps Involved
## 1. Environment Setup:

Install the necessary libraries, including pymysql, pandas, xgboost, sklearn, and seaborn.

2.Database Connection:

Establish a connection to a MySQL database using pymysql.

Load the Titanic dataset into a Pandas DataFrame using SQL queries.

3. Data Exploration:

Display the first few rows of the dataset to understand its structure.

Check for null values in the dataset and print the count of missing values per column.

4. Data Preprocessing:

Use LabelEncoder to convert categorical variables (sex and embarked) into numerical format.

Select relevant features for the model: pclass, sex, age, sibsp, parch, fare, and embarked.

Split the dataset into training and testing sets using train_test_split.

5. Model Training:

Convert the training and testing sets into XGBoost's DMatrix format, which efficiently handles missing values.

Define the parameters for the XGBoost model, including the objective function, evaluation metric, learning rate, and tree depth.

Train the model using the training data, with early stopping to prevent overfitting.

6. Model Prediction:

Use the trained model to predict survival probabilities on the test set.

Convert probabilities to binary predictions (0 or 1) based on a threshold of 0.5.

7. Model Evaluation:

Calculate and print the accuracy of the model.

Generate and display a classification report that includes precision, recall, and F1-score.

8. Model Saving:

Save the trained model to a file (xgboost.pkl) using the pickle library for future use.

# Conclusion
This project demonstrates the end-to-end process of building a machine learning model for binary classification using the Titanic dataset. By leveraging XGBoost, the project showcases effective data handling, model training, and evaluation techniques, providing insights into the factors influencing survival on the Titanic. The saved model can be utilized for further analysis or deployment in applications requiring survival predictions.
