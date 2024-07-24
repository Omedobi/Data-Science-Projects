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
with open('src/config.json', 'r') as config_file:
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

def get_db_connection():
    """Get the database connection"""
    return sqlite3.connect('database/EmissionDatabase.db')

def fetch_data_from_db(columns):
    """Fetch data from the database based on provided feature list. """
    with get_db_connection() as conn:
        columns_escaped = [f"`{col}`" for col in columns]
        query = f"SELECT {','.join(columns_escaped)} FROM bard_data"
        try:
            data = pd.read_sql_query(query, conn)
            st.success('Data fetched successfully. Please proceed to the next step.')
            return pd.DataFrame(data)            
        except pd.io.sql.DatabaseError as e:
            logging.error(f"SQL Error: {e}")
            st.error(f'Failed to fetch data. Please check the database and query.')


def preprocess_data(user_data, feature_list):
    """Preprocess user data by encoding categorical variables, scaling and removing duplicates."""
    if user_data.empty:
        st.warning('No data to preprocess.')
        return pd.DataFrame()
    
    label_encoders = {}
    for column in user_data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        user_data[column] = le.fit_transform(user_data[column])
        label_encoders[column] = le 

    scaler = StandardScaler()
    user_data = user_data.loc[:, feature_list].drop_duplicates()
    user_data_scaled = scaler.fit_transform(user_data)
    return pd.DataFrame(user_data_scaled, columns=user_data.columns)

def setup_sidebar():
    """Set up sidebar for user input and configuration."""
    st.title("Analyzing Vehicle Emissions: Predicting, Clustering, and Detecting Anomalies")
    
    content = """ 
    ### Analyzing Vehicle Emissions \n
    The automotive industry faces growing pressure to minimize its environmental footprint, especially regarding CO2 emissions. 
    Effective management and comprehension of vehicle emissions are vital for manufacturers, regulators, and consumers. 
    This report presents an in-depth analysis focused on predicting CO2 emissions, grouping vehicles by their emissions and fuel efficiency, 
    and identifying anomalies in vehicle emissions. I will outline the objectives of each task and explain how they enhance our understanding of vehicle emissions.

    #### Predicting CO2 Emission (Regression)
    The primary goal of this task is to accurately predict the real-world CO2 emissions (grams per mile) of a vehicle. 
    This prediction leverages various vehicle features, including:
    - **Weight**: The overall mass of the vehicle.
    - **Engine Displacement**: The total volume of all the cylinders in the engine.
    - **Horsepower**: The engine's power output.
    - **Miles Per Gallon (MPG)**: Fuel efficiency metrics including city, highway, and combined MPG.

    By utilizing these features, the regression model aims to provide insights into the environmental 
    impact of vehicles. This can help manufacturers design more efficient vehicles, assist regulators in 
    setting standards, and enable consumers to make informed choices. Accurate prediction of CO2 emissions is 
    not just a technical challenge but a step towards sustainable mobility.

    #### Clustering Vehicles Based on Emissions and Fuel Efficiency
    The goal of this task is to group vehicles into clusters based on their CO2 emissions and fuel efficiency metrics. Clustering helps in:
    - Identifying patterns and similarities among different vehicles.
    - Classifying vehicles into distinct groups that share common characteristics regarding their emissions and fuel efficiency.
    - Providing insights into the performance and environmental impact of different vehicle types.

    By clustering vehicles, we can better understand market segments and develop targeted strategies for improvement. 
    For instance, clusters of high-emission vehicles might be targeted for design overhauls, while low-emission clusters could 
    highlight successful technologies and practices worth replicating.
             
    #### Anomaly Detection in Vehicle Emissions
    This task aims to detect vehicles with unusually high or low CO2 emissions compared to similar vehicles. The specific objectives include:
    - Identifying outliers in the dataset that may indicate data quality issues, unusual vehicle behavior, or potential regulatory violations.
    - Highlighting vehicles that significantly deviate from the norm, prompting further investigation and corrective actions.
    - Ensuring the reliability and accuracy of emissions data, which is critical for regulatory compliance and environmental monitoring.

    Anomaly detection is vital for maintaining data integrity and identifying vehicles that may require special attention due to their deviation from expected performance.
    """
    st.markdown(f"""
        <div style='height: 400px; overflow-y: scroll; padding: 10px; border: 1px solid #ddd; border-radius: 10px;'>
            {content}
        """, unsafe_allow_html=True)
    st.sidebar.title('Analyzing Vehicle Emissions')
    analysis_type = st.sidebar.selectbox('Select analysis', ('Anomaly Detection', 'Clustering', 'Prediction'))
    return analysis_type

def display_analysis(option, user_data):
    """Display the selected analysis type and handle user data."""
    if user_data.empty:
        st.error('No data provided for analysis')
        return
    
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
        st.success(f'Number of anomalies detected: {len(anomalies)}')
        st.dataframe(anomalies)
    else:
        st.error("Anomaly detection model is not loaded or user data is empty.")

def clustering(user_data):
    """Function to handle clustering."""
    cluster_model = models.get('cluster')
    if (cluster_model is not None) and (len(user_data) > 0):
        predictions = cluster_model.fit_predict(user_data)
        
    # Binning the cluster
        cluster_bin = {
    0:'Low Emission',
    1:'Medium Emission',
    2:'High Emission',
    3:'Very High Emission'
}
        user_data['Cluster'] = predictions
        user_data['Cluster_bin'] = user_data['Cluster'].map(cluster_bin)
        
        cluster_count = user_data['Cluster_bin'].value_counts().to_dict()
        cluster_summary = ", ".join([f"{cluster}: {count}" for cluster, count in cluster_count.items()])
        st.success(f'Cluster results: \n{cluster_summary}')
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
        st.success('CO2 Emission Predictions:')
        st.dataframe(user_data)
    else:
        st.error("Prediction model is not loaded or user data is empty.")
        
def perform_eda():
    """Perform EDA using the visualization script"""
    EdaList = config['features'].get('EDA', [])
    if not EdaList:
        st.error('No feature list defined for EDA in the configuration')
        return
    data = fetch_data_from_db(EdaList)
    if data.empty:
        st.error('No data available for EDA')
        return
    
    eda = EDA(data)
    eda.show_plots()
       
def show_visualization(option):
    """Function to show visualization options."""
    if option in ['Anomaly Detection', 'Clustering', 'Prediction']:
        st.write(f"### {option} Visualizations")
        if option == 'Anomaly Detection':
            st.image('image/uMAP-plot.png', caption='Anomaly uMAP plot', use_column_width=True)
        elif option == 'Clustering':
            st.image('image/Elbow-curve-cluster.png', caption='Elbow Curve Plot', use_column_width=True)
            st.image('image//Gmm-cluster.png', caption='Cluster Plot', use_column_width=True)
        elif option == 'Prediction':
            st.image('image/regressor-evaluation.png', caption='Prediction Evaluation', use_column_width=True)
            st.image('image/shap_xgb.png', caption='shap Feature Importance plot', use_column_width=True)

def main():
    
    option = setup_sidebar()
    conn = get_db_connection()
    try:
        if st.sidebar.button('Fetch Data'):
            feature_list = config["features"].get(option, [])
            user_data = fetch_data_from_db(feature_list)
            st.session_state['user_data'] = user_data            
            st.snow()
            
        if 'user_data' in st.session_state:
            if st.sidebar.button('Perform analysis'):
                with st.container():
                    display_analysis(option, st.session_state['user_data'])       
                    
            if st.sidebar.button('Perform EDA'):
                with st.container():
                    perform_eda()
                    show_visualization(option)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()


