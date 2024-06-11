import streamlit as st
import pandas as pd
import pickle
import logging
import numpy as np
from sklearn.model_selection import train_test_split

# @st.cache_resource
# def load_data():
#     data = pd.read_csv('data\diabetes.csv')
#     return data
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_models():
    model_paths = [
        'saved_model/log_model.pkl',
        'saved_model/xgb_model.pkl'
    ]
    models = []
    for path in model_paths:
        try:
            with open(path, 'rb') as file:
                models.append(pickle.load(file))
            logging.info(f'Model loaded successfully from {path}')
        except Exception as e:
            logging.error(f"Failed to load model from {path}: {e}")
            raise e
    return tuple(models)

log_model, xgb_model = load_models()

    
# log_model, xgb_model = load_models()

# Define the user interface
st.title("Diabetes Prediction Application")

# if st.button('Load and Prepare Data'):
#     data = load_data()
#     features = data.drop(columns=['Outcome', 'SkinThickness'], axis=1)
#     target = data['Outcome']
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

#     # Display dataset information
#     st.write(f'Training set: {X_train.shape}, {y_train.shape}')
#     st.write(f'Testing set: {X_test.shape}, {y_test.shape}')

#Model Selection
options = ['Select a model...', 'Logistic Regression','XGBoost Classifier']
model_option = st.selectbox('Choose a model for prediction:', options)
if model_option == 'Choose a model':
    st.warning('Please select a model to proceed with the prediction.')
    
else:
    # Add input widgets for user input
    st.sidebar.header('Features')
    Pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 117)
    BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    Insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    SkinThickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)  # If needed
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)

 
# Create a DataFrame with user
    data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age],
    })

# Make predictions
if st.button('Predict') and model_option != 'Select a model...':
    if model_option == 'Logistic Regression':
        model = log_model
    else:
        model = xgb_model
        
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)
    
    # Display prediction
    st.write('### Prediction')
    if prediction[0] == 1:
        st.write('The patient is predicted to have diabetes.')
    else:
        st.write('The patient is predicted to be diabetes-free.')
   
    # Display prediction probabilities     
    st.write('### Prediction Probabilities')
    st.write(f'Probability of diabetes: {prediction_proba[0][1]:.2f}')
    st.write(f'Probability of no diabetes: {prediction_proba[0][0]:.2f}')