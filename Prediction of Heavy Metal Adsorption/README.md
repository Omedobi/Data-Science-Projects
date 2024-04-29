
![Waste-water](img/waste_water.jpg)

## PREDICTION OF HEAVY METALS ADSORPTION ON ACTIVATED CARBON FROM COCONUT SHELLS

This repository contains code for analyzing and predicting the adsorption efficiency of activated carbon derived from coconut shells for removing heavy metal contaminants such as lead (Pb), zinc (Zn), and cadmium (Cd) from water sources. The analysis involves exploring various parameters including pH, temperature, adsorption capacity, and time to understand their impact on the adsorption process.

### Overview of the Repository

1. **Data Merging and Preprocessing**: The repository starts by loading datasets containing information on different experimental parameters such as pH, temperature, adsorption dose, and time. These datasets are merged and preprocessed to create comprehensive datasets for further analysis.

2. **Exploratory Data Analysis (EDA)**: EDA techniques are applied to understand the relationships and patterns within the data. This includes computing descriptive statistics, visualizing data distributions, and analyzing correlations between parameters and contaminants.

3. **Feature Engineering**: The repository demonstrates feature engineering techniques to prepare the data for modeling. This involves splitting the data into input features and target variables, and performing outlier removal using Z-score.

4. **Modeling**: XGBoost regression and Random Forest Regression models were trained to predict the adsorption efficiency of activated carbon for each heavy metal contaminant based on the experimental parameters. Hyperparameter tuning is performed using Bayesian optimization and GridSearchCV to optimize model performance.

5. **Model Evaluation**: The trained models are evaluated using metrics such as mean squared error (MSE), mean absolute error (MAE), and R-squared (R2) score to assess their predictive performance.

6. **Feature Importance and Visualization**: The repository visualizes the feature importances of the input variables in predicting adsorption efficiency for each heavy metal contaminant using XGBoost regression models.

# Model Evaluation

**XGB Regression Model Evaluation**
The code evaluates the performance of the XGB regression models by computing several evaluation metrics, including Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) score for each dataset. These metrics provide insights into how well the models are performing in predicting the target variables.

### Steps:

1. Import necessary libraries for evaluation metrics from `sklearn`.
2. Calculate evaluation metrics (MSE, MAE, R²) for each dataset using the trained XGB regression models.
3. Print the evaluation results for each dataset.
4. Compile evaluation metrics into a DataFrame for better visualization.

### XGB Regression Feature Importances and SHAP Value

The code also analyzes the feature importances of the input variables in predicting adsorption efficiency for each heavy metal contaminant using XGB regression models. Additionally, it utilizes SHAP (SHapley Additive exPlanations) values to explain the output of the models.

### Steps:

1. Calculate feature importances for each dataset's XGB regression model.
2. Sort the features based on their importances.
3. Visualize the feature importances using bar plots for each dataset.
4. Plot Actual vs Predicted values to visually inspect the performance of the models.
5. Calculate SHAP values and generate summary plots for each dataset to understand the impact of features on model predictions.

### Time Taken for SHAP Value Calculation

The code also measures the time taken to compute SHAP values for each dataset using the XGB regression models. This provides insights into the computational efficiency of the SHAP value calculation process.


**RandomForest Regression model Evaluation**

This code evaluates the performance of RandomForest regression models trained on different datasets (`temp_df`, `time_df`, `ads_df`, `pH_df`). The evaluation includes calculating Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) score for each dataset. The evaluation metrics help assess how well the models are performing in predicting the target variables.

### Steps:

1. Import necessary libraries for evaluation metrics from sklearn.
2. Calculate evaluation metrics (MSE, MAE, R²) for each dataset using the trained RandomForest regression models.
3. Print the evaluation results for each dataset.
4. Compile evaluation metrics into a DataFrame for better visualization.

## Features Importance and Shap value

This part of the code analyzes the importance of features in predicting the target variables using the RandomForest regression models. It also utilizes SHAP (SHapley Additive exPlanations) values to explain the output of the models.

### Steps:

1. Calculate feature importances for each dataset's RandomForest regression model.
2. Sort the features based on their importances.
3. Visualize the feature importances using bar plots for each dataset.
4. Plot Actual vs Predicted values to visually inspect the performance of the models.
5. Calculate SHAP values and generate summary plots for each dataset to understand the impact of features on model predictions.

## Evaluation Results

- The evaluation metrics provide insights into the performance of the RandomForest regression models on different datasets.
- Feature importance analysis helps identify which features are crucial for predicting the target variables.
- SHAP values offer a deeper understanding of how each feature contributes to individual predictions.

These analyses collectively aid in understanding the model's behavior and identifying areas for improvement in the predictive performance.

**Install Dependencies**: Install the required Python dependencies listed in the `requirements.txt` file using:
   ```
   pip install -r requirements.txt
   ```

**Run the Code**: Navigate to the cloned repository directory and run the Python scripts to execute the analysis and modeling steps.

Feel free to contribute to this repository by opening issues or submitting pull requests. If you have any questions or suggestions, please contact me by [Email](mailto:ikennaanywuike@gmail.com).

### License

This project is licensed under the [MIT License](LICENSE).
