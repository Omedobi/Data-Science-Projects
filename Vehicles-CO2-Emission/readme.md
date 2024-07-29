![Vehicle-Emission](image/p1.jpg)

## Table of Contents
1. [Introduction](#introduction)
2. [Predicting CO2 Emission (Regression)](#predicting-co2-emission-regression)
   - [Objective](#objective)
   - [Approach](#approach)
   - [Model Performance](#model-performance)
     - [Mean Squared Error (MSE)](#mean-squared-error-mse)
     - [R-squared (R²) Score](#r-squared-r²-score)
     - [Root Mean Squared Error (RMSE)](#root-mean-squared-error-rmse)
   - [Conclusion](#conclusion)
3. [Clustering Vehicles Based on Emissions and Fuel Efficiency](#clustering-vehicles-based-on-emissions-and-fuel-efficiency)
   - [Objective](#objective-1)
   - [Approach](#approach-1)
   - [Model Performance](#model-performance-1)
     - [Silhouette Score](#silhouette-score)
     - [Calinski-Harabasz Index](#calinski-harabasz-index)
     - [Davies-Bouldin Index](#davies-bouldin-index)
   - [Conclusion](#conclusion-1)
4. [Anomaly Detection in Vehicle Emissions](#anomaly-detection-in-vehicle-emissions)
   - [Objective](#objective-2)
   - [Approach](#approach-2)
   - [Conclusion](#conclusion-2)
5. [General Conclusion](#general-conclusion)
6. [Setup and Dependencies](#setup-and-dependencies)
   - [Prerequisites](#prerequisites)
   - [Setting Up Dependencies](#setting-up-dependencies)
     - [Clone the Repository](#clone-the-repository)
     - [Create a Virtual Environment](#create-a-virtual-environment)
     - [Install Dependencies](#install-dependencies)
   - [Docker Setup](#docker-setup)
     - [Build the Docker Image](#build-the-docker-image)
     - [Run the Docker Container](#run-the-docker-container)
     - [Access the Jupyter Notebook](#access-the-jupyter-notebook)
   - [Dockerfile](#dockerfile)
   - [requirements.txt](#requirementstxt)

---

# Introduction

This project involves three main tasks related to vehicle emission data analysis: predicting CO2 emissions, clustering vehicles based on emissions and fuel efficiency, and detecting anomalies in vehicle emissions. Each task is crucial for understanding and managing vehicle emissions, improving data quality, and ensuring regulatory compliance.

## Predicting CO2 Emission (Regression)

### Objective
The primary objective of this task is to predict the real-world CO2 emissions (g/mi) of a vehicle based on various features such as weight, engine displacement, horsepower, and MPG. Accurate predictions of CO2 emissions are essential for assessing environmental impact and complying with emission regulations.

### Approach
I employed two regression algorithms, XGBRegressor and RandomForestRegressor, to predict CO2 emissions. The models were trained and evaluated using a dataset containing vehicle features and corresponding CO2 emission values. The performance of these models was assessed using three key metrics: Mean Squared Error (MSE), R-squared (R²) score, and Root Mean Squared Error (RMSE).

### Model Performance

#### Mean Squared Error (MSE)
- **XGBRegressor**: 552.803245
- **RandomForestRegressor**: 1556.085791

MSE measures the average squared difference between the actual and predicted values. Lower MSE values indicate better model performance. The XGBRegressor achieved a significantly lower MSE compared to the RandomForestRegressor, indicating more accurate predictions.

#### R-squared (R²) Score
- **XGBRegressor**: 0.915434
- **RandomForestRegressor**: 0.761956 

R² score represents the proportion of variance in the dependent variable that can be explained by the independent variables. Higher R² values indicate better model fit. The XGBRegressor's R² score of 0.915434 suggests it explains approximately 92% of the variance in CO2 emissions, outperforming the RandomForestRegressor.

#### Root Mean Squared Error (RMSE)
- **XGBRegressor**: 23.511768
- **RandomForestRegressor**: 39.447253

RMSE is the square root of MSE, providing a measure of the average magnitude of prediction errors. Lower RMSE values indicate better performance. The XGBRegressor's RMSE of 23.511768 was significantly lower than the RandomForestRegressor's, reinforcing the XGBRegressor's superior prediction accuracy.

### Conclusion
The XGBRegressor outperformed the RandomForestRegressor across all evaluation metrics, making it the recommended algorithm for predicting CO2 emissions from vehicles. Its lower MSE and RMSE values, along with a higher R² score, indicate more accurate predictions and better model fit.

---

## Clustering Vehicles Based on Emissions and Fuel Efficiency

### Objective
The goal of this task is to group vehicles into clusters based on their CO2 emissions and fuel efficiency metrics. Clustering helps identify patterns and similarities among vehicles, enabling better understanding and management of vehicle emissions.

### Approach
I evaluated three clustering algorithms: Agglomerative Clustering, Gaussian Mixture, and KMeans. The performance of these algorithms was assessed using the Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index, which measure cluster cohesion and separation.

### Model Performance

#### Silhouette Score
- **Agglomerative Clustering**: 0.188
- **Gaussian Mixture**: 0.266
- **KMeans**: 0.204

The Silhouette Score measures the compactness and separation of clusters. Higher values indicate better-defined clusters. The Gaussian Mixture algorithm achieved the highest Silhouette Score, suggesting well-separated and cohesive clusters.

#### Calinski-Harabasz Index
- **Agglomerative Clustering**: 912.691
- **Gaussian Mixture**: 813.698
- **KMeans**: 1016.065

The Calinski-Harabasz Index evaluates the ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate better clustering performance. The KMeans algorithm had the highest index value, indicating the best performance in this metric.

#### Davies-Bouldin Index
- **Agglomerative Clustering**: 1.347
- **Gaussian Mixture**: 1.262
- **KMeans**: 1.308

The Davies-Bouldin Index measures the average similarity ratio of each cluster with its most similar cluster. Lower values indicate better performance. The Gaussian Mixture algorithm had the lowest index value, suggesting the best clustering performance.

### Conclusion
The Gaussian Mixture algorithm is recommended for clustering vehicles based on their CO2 emissions and fuel efficiency metrics due to its superior performance in the Silhouette Score and Davies-Bouldin Index. The KMeans algorithm also performed well, especially in the Calinski-Harabasz Index. Agglomerative Clustering, while useful, was less effective compared to the other algorithms.

---

## Anomaly Detection in Vehicle Emissions

### Objective
The objective of this task is to detect vehicles with unusually high or low CO2 emissions compared to similar vehicles. Identifying anomalies helps in improving data quality, identifying potential errors, and ensuring regulatory compliance.

### Approach
I used Isolation Forest and K-Nearest Neighbors (KNN) algorithms for anomaly detection. The detected anomalies were analyzed by calculating the median absolute deviation of each feature and identifying the columns with the maximum deviations. This helps in pinpointing the specific features contributing to the anomalies.

### Isolation Forest Anomalies
1. **Calculation of Medians**: The median of each feature was calculated to serve as a reference point.
2. **Deviation Calculation**: Absolute deviations from the median were calculated for each anomaly.
3. **Max Deviation Identification**: Columns with the maximum deviations were identified and analyzed.

### KNN Anomalies
1. **Calculation of Medians**: Similar to the Isolation Forest approach, the median of each feature was calculated.
2. **Deviation Calculation**: Absolute deviations from the median were calculated for each anomaly.
3. **Max Deviation Identification**: Columns with the maximum deviations were identified and analyzed.

### Conclusion
The anomaly detection process using both Isolation Forest and KNN effectively identified specific features with maximum deviations. This helps in pinpointing potential errors or unusual patterns in the data, contributing to better data quality and regulatory compliance.

---

## General Conclusion

This project provides a comprehensive analysis of vehicle emissions, leveraging advanced machine learning techniques to predict CO2 emissions, cluster vehicles based on emissions and fuel efficiency, and detect anomalies. The results from these tasks offer valuable insights for improving vehicle emission data quality, optimizing vehicle performance, and ensuring compliance with environmental regulations.

### Key Takeaways:
- **Prediction Accuracy**: The XGBRegressor demonstrated superior performance in predicting CO2 emissions, providing highly accurate and reliable predictions.
- **Effective Clustering**: The Gaussian Mixture algorithm effectively grouped vehicles into meaningful clusters based on emissions and fuel efficiency, helping identify patterns and similarities among vehicles.
- **Anomaly Detection**: The use of Isolation Forest and KNN algorithms for anomaly detection proved effective in identifying unusual patterns in vehicle emissions data, enhancing data quality and reliability.

### Significance
The methodologies and findings from this project have significant implications for the automotive industry and environmental agencies. By accurately predicting emissions, identifying patterns, and detecting anomalies, stakeholders can make informed decisions to reduce emissions, improve vehicle design, and ensure regulatory compliance. The approach and techniques applied in this project can be extended to other domains where similar analysis and insights are required, showcasing the versatility and impact of machine learning in data analysis and decision-making.

---

## Setup and Dependencies

### Prerequisites
Ensure you have the following software installed on your system:
- Python 3.10 or higher
- Git
- Docker

### Setting Up Dependencies

#### Clone the Repository
```bash
git clone https://github.com/Omedobi/Data-Science-Projects/tree/main/Vehicles-CO2-Emission.git
cd Vehicle-CO2-Emission
```

#### Create a Virtual Environment
```bash
python3 -m venv myenv
source myenv/Scripts/activate  # On Windows, 
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### Docker Setup

#### Start the application by running
```bash
docker compose up --build```

#### Build the Docker Image
```bash
docker build -t co2-emission .
```

#### Run the Docker Container
```bash
docker run -p 8000:8000 co2-emission
```

# Run the Streamlit application
```bash
CMD ["streamlit", "run", "__main__.py", "--server.port=8000"]
```

By following the setup instructions, you can recreate the environment used for this analysis and further explore the code and findings.