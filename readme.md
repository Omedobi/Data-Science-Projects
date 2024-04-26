
## Loan Approval Prediction Model

This repository presents a machine learning model designed to predict loan approval status based on various applicant attributes. Here's a comprehensive overview of the project:

### Installation

To get started, ensure you have the required dependencies installed. You can do this by running:

```bash
pip install -r Requirements.txt
```

### Dataset Overview

The dataset used contains essential information about loan applicants, including their demographics, financial details, and loan status. Here's a summary of the dataset columns:

- **loan_id**: Unique identifier for each loan application
- **no_of_dependents**: Number of dependents of the applicant
- **education**: Education level of the applicant
- **self_employed**: Indicates if the applicant is self-employed
- **income_annum**: Annual income of the applicant
- **loan_amount**: Amount of loan requested by the applicant
- **loan_tenure**: Duration of the requested loan in years
- **credit_score**: Credit score of the applicant
- **residential_asset_value**: Value of the applicant's residential assets
- **commercial_asset_value**: Value of the applicant's commercial assets
- **luxury_asset_value**: Value of the applicant's luxury assets
- **bank_assets_value**: Value of the applicant's bank assets
- **loan_status**: Status of the loan application (Approved/Rejected)

### Data Preprocessing

The data preprocessing phase involves cleaning and preparing the dataset for model training. This includes handling missing values, encoding categorical variables, and exploring relationships between features and the target variable.

### Model Development

Three powerful machine learning algorithms were utilized for loan approval prediction:

1. **Random Forest Classifier**
2. **XGBoost Classifier**
3. **CatBoost Classifier**

Hyperparameter tuning techniques such as GridSearchCV and BayesSearchCV were employed to optimize each model's performance.

### Model Evaluation

The performance of each model was assessed using various evaluation metrics, including confusion matrix visualization, classification reports, and key performance indicators such as R2 Score, Mean Squared Error, and Mean Absolute Error.

### Conclusion

After thorough analysis and evaluation, the XGBoost Classifier emerged as the top-performing model, achieving an impressive accuracy of 99%. This model demonstrated superior predictive capabilities compared to the Random Forest and CatBoost classifiers, which achieved accuracies of 98%. The XGBoost Classifier is therefore recommended for predicting loan approval status in this scenario.
