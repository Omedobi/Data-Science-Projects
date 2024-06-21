import streamlit as st
import logging
import pandas as pd
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

@st.cache_resource
def load_models():
    models = {}
    # Load each model
    for model_type, path in config["model_paths"].items():
        try:
            with open(path, 'rb') as file:
                models[model_type] = joblib.load(file)
            logging.info(f'{model_type.capitalize()} model loaded successfully from {path}')
        except Exception as e:
            logging.error(f'Failed to load {model_type} model from {path}: {e}')
            raise e
    return models

# Load models
models = load_models()

@st.cache_data
def load_test_data():
    return pd.read_csv(config['test_data_path'])

test_data = load_test_data()

# Sidebar navigation
st.sidebar.title('Vehicle Emission Analysis')
option = st.sidebar.selectbox('Select Analysis', ('Anomaly Detection', 'Clustering', 'Prediction'))

def classify_model_year(year):
    if year in range(1975, 1999):
        return "Old_model"
    elif year in range(2000, 2010):
        return "Modern_model"
    elif year in range(2011, 2023):
        return "Recent_model"
    else:
        return "Others"

def get_user_input():
    st.sidebar.subheader('Input vehicle data')
    model_year_options = [1975, 2000, 2011, 2024]
    model_year_labels = [classify_model_year(year) for year in model_year_options]
    model_year = st.sidebar.selectbox('Model Year', model_year_options, format_func=lambda x: model_year_labels[model_year_options.index(x)])
    
    data = {
        'Manufacturer': st.sidebar.selectbox('Manufacturer', ('Ford', 'Hyundai', 'Kia', 'GM', 'Honda', 'BMW', 'Volkswagen', 'Toyota', 'Nissan', 'Tesla', 'Stellantis', 'Mazda', 'Mercedes', 'Subaru')),
        'Model_Year': model_year,
        'Regulatory_Class': st.sidebar.selectbox('Regulatory Class', ('Car', 'Truck')),
        'Vehicle_Type': st.sidebar.selectbox('Vehicle Type', ('Car SUV', 'Minivan/Van', 'Pickup', 'Sedan/Wagon', 'Truck SUV')),
        '2_Cycle_MPG': st.sidebar.slider('2 Cycle MPG', min_value=9.0, max_value=180.0, step=0.1),
        'Weight_(lbs)': st.sidebar.slider('Weight (lbs)', min_value=2000.0, max_value=7000.0, step=1.0),
        'Footprint_(sq._ft.)': st.sidebar.slider('Footprint (sq. ft.)', min_value=41.0, max_value=69.0, step=0.1),
        'Engine_Displacement': st.sidebar.slider('Engine Displacement', min_value=79.0, max_value=371.0, step=0.1),
        'Horsepower_(HP)': st.sidebar.slider('Horsepower (HP)', min_value=53.0, max_value=753.0, step=1.0),
        'Acceleration_(0_60_time_in_seconds)': st.sidebar.slider('Acceleration (0-60 time in seconds)', min_value=2.0, max_value=30.0, step=0.1),
        'Drivetrain_Front': st.sidebar.selectbox('Drivetrain Front', ['No', 'Yes']),
        'Drivetrain_4WD': st.sidebar.selectbox('Drivetrain 4WD', ['No', 'Yes']),
        'Drivetrain_Rear': st.sidebar.selectbox('Drivetrain Rear', ['No', 'Yes']),
        'Transmission_Manual': st.sidebar.selectbox('Transmission Manual', ['No', 'Yes']),
        'Transmission_Automatic': st.sidebar.selectbox('Transmission Automatic', ['No', 'Yes']),
        'Transmission_Lockup': st.sidebar.selectbox('Transmission Lockup', ['No', 'Yes']),
        'Transmission_CVT_(Hybrid)': st.sidebar.selectbox('Transmission CVT (Hybrid)', ['No', 'Yes']),
        'Fuel_Delivery_Carbureted': st.sidebar.selectbox('Fuel Delivery Carbureted', ['No', 'Yes']),
        'Fuel_Delivery_Gasoline_Direct_Injection_(GDI)': st.sidebar.selectbox('Fuel Delivery Gasoline Direct Injection (GDI)', ['No', 'Yes']),
        'Fuel_Delivery_Port_Fuel_Injection': st.sidebar.selectbox('Fuel Delivery Port Fuel Injection', ['No', 'Yes']),
        'Fuel_Delivery_Throttle_Body_Injection': st.sidebar.selectbox('Fuel Delivery Throttle Body Injection', ['No', 'Yes']),
        'Powertrain_Diesel': st.sidebar.selectbox('Powertrain Diesel', ['No', 'Yes']),
        'Powertrain_Electric_Vehicle_(EV)': st.sidebar.selectbox('Powertrain Electric Vehicle (EV)', ['No', 'Yes']),
        'Powertrain_Plug_in_Hybrid_Electric_Vehicle_(PHEV)': st.sidebar.selectbox('Powertrain Plug-in Hybrid Electric Vehicle (PHEV)', ['No', 'Yes']),
        'Powertrain_Fuel_Cell_Vehicle_(FCV)': st.sidebar.selectbox('Powertrain Fuel Cell Vehicle (FCV)', ['No', 'Yes']),
        'Powertrain_Other_(incl._CNG)': st.sidebar.selectbox('Powertrain Other (incl. CNG)', ['No', 'Yes']),
        'Powertrain_Gasoline_Hybrid': st.sidebar.selectbox('Powertrain Gasoline Hybrid', ['No', 'Yes']),
        'Powertrain_Gasoline': st.sidebar.selectbox('Powertrain Gasoline', ['No', 'Yes']),
        'Turbocharged_Engine': st.sidebar.selectbox('Turbocharged Engine', ['No', 'Yes']),
        'Multivalve_Engine': st.sidebar.selectbox('Multivalve Engine', ['No', 'Yes']),
        'Variable_Valve_Timing': st.sidebar.selectbox('Variable Valve Timing', ['No', 'Yes']),
        'Transmission_CVT_(Non_Hybrid)': st.sidebar.selectbox('Transmission CVT (Non-Hybrid)', ['No', 'Yes']),
        'Gears': st.sidebar.slider('Gears', min_value=1, max_value=10, step=1),
    }
    return pd.DataFrame(data, index=[0])

user_data = get_user_input()
preprocessor = models['preprocessor']
user_data_preprocessed = preprocessor.transform(user_data)
test_data_preprocessed = preprocessor.transform(test_data)

def anomaly_detection():
    st.title('Anomaly Detection in Vehicle Emissions')
    if st.sidebar.button('Detect Anomalies'):
        anom_model = models['anomaly']
        predictions = anom_model.predict(user_data_preprocessed)
        anomalies = user_data[predictions == -1]
        st.write(f'Number of anomalies detected: {len(anomalies)}')
        st.dataframe(anomalies)
        st.subheader('Mechanical Insights')
        st.write("Provide mechanical analysis based on detected anomalies.")
        st.subheader('Environmental Insights')
        st.write("Provide environmental analysis based on detected anomalies.")

def clustering():
    st.title('Clustering Vehicles Based on Emission and Fuel Efficiency')
    if st.sidebar.button('Cluster Vehicles'):
        cluster_model = models['cluster']
        predictions = cluster_model.fit_predict(user_data_preprocessed)
        user_data['Cluster'] = predictions
        st.write('Cluster prediction')
        st.dataframe(user_data)
        st.subheader('Mechanical Cluster Analysis')
        st.write("Provide mechanical cluster analysis.")
        st.subheader('Environmental Cluster Analysis')
        st.write("Provide environmental cluster analysis.")

def prediction():
    st.title('Predicting CO2 Emission')
    if st.sidebar.button('Predict CO2 Emissions'):
        reg_model = models['regression']
        predictions = reg_model.predict(test_data_preprocessed)
        test_data['CO2 Emission Prediction'] = predictions
        st.write('CO2 Emission Predictions:')
        st.dataframe(test_data[['CO2 Emission Prediction']])
        st.subheader('Mechanical Prediction Insights')
        st.write("Provide mechanical insights based on CO2 emission predictions.")
        st.subheader('Environmental Prediction Insights')
        st.write("Provide environmental insights based on CO2 emission predictions.")

if option == 'Anomaly Detection':
    anomaly_detection()
elif option == 'Clustering':
    clustering()
elif option == 'Prediction':
    prediction()
