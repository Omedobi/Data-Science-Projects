import streamlit as st
import os
import pandas as pd
import logging
import joblib
import json
import pycaret
import sqlite3
from sklearn.preprocessing import StandardScaler, LabelEncoder


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
conn = sqlite3.connect('C:/Users/admin/EmissionDatabase.db')
conn.row_factory = sqlite3.Row

def list_tables():
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    return tables

def fetch_data_from_db(columns):
    columns_escaped = [f"`{col}`" for col in columns]
    query = f"SELECT {','.join(columns_escaped)} FROM bard_data"
    data = pd.read_sql_query(query, conn)
    data.columns = columns
    return data

def setup_sidebar():
    """Set up sidebar for user input and configuration."""
    st.sidebar.title('Vehicle Emission Analysis')
    return st.sidebar.selectbox('Select analysis', ('Anomaly Detection', 'Clustering', 'Prediction'))

def preprocess_data(user_data, feature_list):
    """Preprocess user data by encoding categorical variables, scaling and removing duplicates."""
    label_encoders = {}
    for column in user_data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        user_data[column] = le.fit_transform(user_data[column])
        label_encoders[column] = le
    
    user_data = user_data[feature_list]
    user_data = user_data.loc[:, ~user_data.columns.duplicated()]  

    scaler = StandardScaler()
    user_data_scaled = scaler.fit_transform(user_data)
    user_data_scaled = pd.DataFrame(user_data_scaled, columns=user_data.columns)
    return user_data_scaled

def display_analysis(option, user_data):
    """Display the selected analysis type and handle user data."""
    feature_list = config["features"].get(option)
    if not feature_list:
        st.error(f"Feature list for {option} not found in configuration.")
        return

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
        
def show_visualization():
    """Function to show visualization options."""
    st.sidebar.write("Visualization")
    visualization_option = st.sidebar.selectbox('Show visualization', ('Elbow curve plot', 'Cluster plot'))
    if st.sidebar.button('Show visualization'):
        if visualization_option == 'Elbow curve plot':
            st.image('C:/Users/admin/Documents/Conda files/Data Science Projects/CO2-Emission/image/Elbow-curve-cluster.png', caption='Elbow Curve Plot', use_column_width=True)
        elif visualization_option == 'Cluster plot':
            st.image('C:/Users/admin/Documents/Conda files/Data Science Projects/CO2-Emission/image/GMM-cluster.png', caption='Cluster Plot', use_column_width=True)

def main():
    """Main function to orchestrate the app flow."""
    option = setup_sidebar()
    st.sidebar.title("Database schema")
    if st.sidebar.button('List of Tables'):
        tables = list_tables()
        st.write("Tables in Database:")
        st.dataframe(tables)
    
    if st.sidebar.button('Fetch Data from Database'):
        feature_list = config["features"].get(option, [])
        if not feature_list:
            st.error(f"Feature list for {option} not found in the configuration")
            return
        
        user_data = fetch_data_from_db(feature_list)
        if not user_data.empty:
            with st.container():
                display_analysis(option, user_data)
                
        else:
            st.warning('No data found in the database.')
    
    show_visualization()

if __name__ == "__main__":
    try:
        main()
    finally:
        conn.close()  # close the database connection
