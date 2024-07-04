import os
import streamlit as st
import pandas as pd
import logging
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.visualize import EDA


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

@st.cache_resource
def load_models():
    models = {}
    for model_type, path in config["model_paths"].items():
        try:
            with open(path, 'rb') as file:
                models[model_type] = joblib.load(path)
            logging.info(f'{model_type.capitalize()} model loaded successfully from {path}')
        except Exception as e:
            logging.error(f'Failed to load {model_type} model from {path}: {e}')
            st.error(f'Error loading {model_type} model. Please check the logs.')
    return models

models = load_models()

# Database connection is initialized once and reused
def get_db_connection():
    return sqlite3.connect('C:/Users/admin/EmissionDatabase.db')

def fetch_data_from_db(columns):
    conn = get_db_connection()
    columns_escaped = [f"`{col}`" for col in columns]
    query = f"SELECT {','.join(columns_escaped)} FROM bard_data"
    try:
        data = pd.read_sql_query(query, conn)
        data.columns = columns
        return data
    finally:
        conn.close()

def preprocess_data(user_data, feature_list):
    """Preprocess user data by encoding categorical variables, scaling and removing duplicates."""
    label_encoders = {}
    for column in user_data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        user_data[column] = le.fit_transform(user_data[column])
        label_encoders[column] = le 

    scaler = StandardScaler()
    user_data = user_data[feature_list].drop_duplicates()
    user_data_scaled = scaler.fit_transform(user_data)
    return pd.DataFrame(user_data_scaled, columns=user_data.columns)

def setup_sidebar():
    """Set up sidebar for user input and configuration."""
    st.sidebar.title('Vehicle Emission Analysis')
    return st.sidebar.selectbox('Select analysis', ('Anomaly Detection', 'Clustering', 'Prediction'))


def display_analysis(option, user_data):
    """Display the selected analysis type and handle user data."""
    if not user_data.empty:
        feature_list = config["features"].get(option)
        user_data_processed = preprocess_data(user_data, feature_list)
        if option == 'Anomaly Detection':
            anomaly_detection(user_data_processed)
        elif option == 'Clustering':
            clustering(user_data_processed)
        elif option == 'Prediction':
            prediction(user_data_processed, feature_list)

def anomaly_detection(user_data):
    """Function to handle anomaly detection."""
    anom_model = models.get('anomaly')
    if anom_model and len(user_data) > 0:
        predictions = anom_model.predict(user_data)
        user_data['Anomalies'] = predictions
        anomalies = user_data[user_data['Anomalies'] == 1].round(2)
        st.write(f'Number of anomalies detected: {len(anomalies)}')
        st.dataframe(anomalies)
    else:
        st.error("Anomaly detection model is not loaded or user data is empty.")

def clustering(user_data):
    """Function to handle clustering."""
    cluster_model = models.get('cluster')
    if (cluster_model is not None) and (len(user_data) > 0):
        predictions = cluster_model.fit_predict(user_data)
        user_data['Cluster'] = predictions
        st.write('Cluster prediction results:')
        st.dataframe(user_data)
    else:
        st.error("Clustering model is not loaded or user data is empty.")

def prediction(user_data, feature_list):
    """Function to handle prediction."""
    reg_model = models.get('regression')
    if reg_model and len(user_data) > 0:
        if reg_model.n_features_in_ != user_data.shape[1]:
            st.error(f"Feature shape mismatch: Model expects {reg_model.n_features_in_} features, but got {user_data.shape[1]}.")
            st.write("Features expected:")
            st.write(feature_list)
            st.write('Features received:')
            st.write(user_data.columns.tolist())
            return
        predictions = reg_model.predict(user_data)
        user_data['CO2 Emission Prediction'] = predictions
        st.write('CO2 Emission Predictions:')
        st.dataframe(user_data)
    else:
        st.error("Prediction model is not loaded or user data is empty.")
        
def perform_eda(user_data):
    """Perform EDA using the visualization script"""
    eda = EDA(user_data)
    eda.show_plots()
       
def show_visualization(option):
    """Function to show visualization options."""
    if option in ['Clustering', 'Prediction']:
        st.write(f"### {option} Visualizations")
        if option == 'Clustering':
            st.image('C:/Users/admin/Documents/Conda files/Data Science Projects/CO2-Emission/image/Elbow-curve-cluster.png', caption='Elbow Curve Plot', use_column_width=True)
            st.image('C:/Users/admin/Documents/Conda files/Data Science Projects/CO2-Emission/image/GMM-cluster.png', caption='Cluster Plot', use_column_width=True)
        elif option == 'Prediction':
            st.image('C:/Users/admin/Documents/Conda files/Data Science Projects/CO2-Emission/image/regressor-evaluation.png', caption='Prediction Evaluation', use_column_width=True)
            st.image('C:/Users/admin/Documents/Conda files/Data Science Projects/CO2-Emission/image/shap_xgb.png', caption='Prediction Evaluation', use_column_width=True)

def main():
    option = setup_sidebar()
    if st.sidebar.button('Fetch Data'):
        feature_list = config["features"].get(option, [])
        user_data = fetch_data_from_db(feature_list)
        st.session_state['user_data'] = user_data
        st.write('Data fetched successfully. Please proceed to the next step.')
        
    if 'user_data' in st.session_state:
        if st.sidebar.button('Perform analysis'):
            with st.container():
                display_analysis(option, st.session_state['user_data'])
                
        if st.sidebar.button('Perform EDA'):
            with st.container():
                perform_eda(st.session_state['user_data'])
                show_visualization(option)

if __name__ == "__main__":
    main()


