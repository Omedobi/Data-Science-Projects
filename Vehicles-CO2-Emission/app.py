import streamlit as st
import pandas as pd
import logging
import joblib
import json
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
    return data

def setup_sidebar():
    """Set up sidebar for user input and configuration."""
    st.sidebar.title('Vehicle Emission Analysis')
    return st.sidebar.selectbox('Select Analysis Type', ('Anomaly Detection', 'Clustering', 'Prediction'))

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
        prediction(user_data_processed)

def anomaly_detection(user_data):
    """Function to handle anomaly detection."""
    anom_model = models.get('anomaly')
    if anom_model and len(user_data) > 0:
        predictions = anom_model.predict(user_data)
        anomalies = user_data[predictions == 1]
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

def prediction(user_data):
    """Function to handle prediction."""
    reg_model = models.get('regression')
    if reg_model and len(user_data) > 0:
        predictions = reg_model.predict(user_data)
        user_data['CO2 Emission Prediction'] = predictions
        st.write('CO2 Emission Predictions:')
        st.dataframe(user_data)
    else:
        st.error("Prediction model is not loaded or user data is empty.")

def main():
    """Main function to orchestrate the app flow."""
    option = setup_sidebar()
    st.sidebar.title("Database schema")
    if st.sidebar.button('List of Tables'):
        tables = list_tables()
        st.write("Tables in Database:")
        st.dataframe(tables)
    
    if st.sidebar.button('Fetch Data from Database'):
        feature_list = config["features"].get(option,[])
        if not feature_list:
            st.error(f"Feature list for {option} not found in the configuration")
            return
        
        user_data = fetch_data_from_db(feature_list)
        if not user_data.empty:
            display_analysis(option, user_data)
        else:
            st.warning('No data found in the database.')
    else:
        st.warning('Click the button to fetch data from the database.')

if __name__ == "__main__":
    try:
        main()
    finally:
        conn.close()  # Ensure the connection is closed when the app exits
