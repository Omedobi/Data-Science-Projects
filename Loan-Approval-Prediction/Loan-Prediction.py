# %% [markdown]
# Install requirements using "pip install -r Requirements.txt"

# %% [markdown]
# ### Data Dictionary
# | Variable | Description |
# | --- | --- |
# |loan_id | Unique loan ID|
# |no_of_dependents | Number of dependents of the applicant|
# |education | Education level of the applicant|
# |self_employed | If the applicant is self-employed or not|
# |income_annum | Annual income of the applicant|
# |loan_amount | Loan amount requested by the applicant|
# |loan_tenure | Tenure of the loan requested by the applicant (in Years)|
# |credit_score | credit score of the applicant|
# |residential_asset_value | Value of the residential asset of the applicant|
# |commercial_asset_value | Value of the commercial asset of the applicant|
# |luxury_asset_value | Value of the luxury asset of the applicant|
# |bank_assets_value | Value of the bank asset of the applicant|
# |loan_status | Status of the loan (Approved/Rejected)|

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import re
import pyarrow
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# %%
loan_data = pd.read_csv("data/loan_approval_dataset.csv")
loan_data.head()

# %%
loan_data.to_parquet('data/loan_approval_dataset.pq')

loan_data.head()

# %% [markdown]
# **Data Cleaning**

# %%
loan_data.columns

# %%
data = loan_data

# %%
data.info()

# %%
data.columns = data.columns.str.strip()

# %%
data.isnull().sum()

# %%
data["Movable_assets"] = data["bank_asset_value"] + data["luxury_assets_value"]
data["Immovable_assets"] = data["residential_assets_value"] + data["commercial_assets_value"]
            

# %%
data.drop(columns=["bank_asset_value","luxury_assets_value","residential_assets_value","commercial_assets_value"], inplace=True, axis=1)

# %%
data.drop("loan_id",axis=1, inplace=True)

# %%
data.rename(columns={"cibil_score":"credit_score"},inplace=True)

# %%
data.columns

# %%
data.isna().sum()

# %%
print(data["education"].value_counts())
print(data["self_employed"].value_counts())
print(data["loan_status"].value_counts())

# %%
data.to_parquet("data/loan_approval_dataset.pq") #ignore this

# %%
data.dtypes

# %%
data.head(2)

# %%
data.isna().sum()


# %%
# data.fillna(data.mean(), inplace=True) #ignore this

# %% [markdown]
# **Exploratory Data Analysis**

# %% [markdown]
# Number of Dependents

# %%
plt.figure(figsize=(8,5))

sns.countplot(x = 'no_of_dependents', data = data,).set_title('Number of Dependents')

plt.tight_layout()
plt.show()

# %% [markdown]
# Education and Income

# %%
fig, ax = plt.subplots(1,2,figsize=(10, 5))
sns.boxplot(x = 'education', y = 'income_annum', data = data, ax=ax[0])
sns.violinplot(x = 'education', y = 'income_annum', data = data, ax=ax[1])

# %% [markdown]
# employment status and education, Loan amount and Tenure

# %%
fig , axes = plt.subplots(1,3, figsize=(17,6))

axes[0].pie(data['self_employed'].value_counts(), labels=['Graduate', 'Not Graduate'], autopct='%1.1f%%', startangle=90)
axes[0].set_title('Education distribution')

sns.countplot(x='self_employed', data = data, hue = 'education', palette='viridis', ax=axes[1]).set_title('Self Employed')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, ha='right', fontsize=16)

sns.lineplot(x = 'loan_term', y = 'loan_amount', data = data, ax=axes[2]).set_title('Loan Amount vs. Loan Term')

for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

plt.tight_layout()
plt.show()

# %% [markdown]
# Score Distribution
# 

# %%
plt.figure(figsize=(8,5))
sns.histplot(data['credit_score'], bins = 40, kde = True, color = 'blue')


# %% [markdown]
# |Credit Score|Meaning|
# |---|---|
# |300-549|Poor|
# |550-649|Fair|
# |650-749|Good|
# |750-799|Very Good|
# |800-900|Excellent|
# 
# Source: [godigit.com](https://www.godigit.com/finance/credit-score/ranges-of-credit-score)

# %% [markdown]
# Asset Distribution

# %%
fig, ax = plt.subplots(1,2,figsize=(8,5))
sns.histplot(data['Movable_assets'], ax=ax[0], color='blue')
sns.histplot(data['Immovable_assets'], ax=ax[1], color='green')

plt.tight_layout()
plt.show()

# %% [markdown]
# Number of Dependants Vs Loan Status

# %%
plt.figure(figsize=(8,5))

sns.countplot(x = 'no_of_dependents', data = data, hue = 'loan_status')

# %% [markdown]
# Education Vs Loan Status

# %%
plt.figure(figsize=(8,5))

sns.countplot(x = 'education', hue = 'loan_status', data = data).set_title('Loan Status by Education')

plt.show()

# %% [markdown]
# Asset Vs Loan Status

# %%
fig, ax = plt.subplots(1,2,figsize=(8,5))
sns.histplot(x  = 'Movable_assets', data = data, ax=ax[0], hue = 'loan_status', multiple='stack')
sns.histplot(x =  'Immovable_assets', data = data, ax=ax[1], hue  = 'loan_status', multiple='stack')

plt.tight_layout()
plt.show()

# %% [markdown]
# Loan Amount and Tenure VS Loan Status

# %%
plt.figure(figsize=(8,5))
sns.lineplot(x='loan_term', y='loan_amount', data=data, hue='loan_status')

plt.show()

# %% [markdown]
# Credit score vs Loan Status

# %%
plt.figure(figsize=(8,5))

sns.violinplot(x='loan_status', y='credit_score', data=data)

plt.show()

# %% [markdown]
# Assets and Income per annum Vs Loan Amount and a boxplot showing No of dependent vs loan amount

# %%
fig, ax = plt.subplots(2,2,figsize=(12, 8))
sns.scatterplot(x='Movable_assets', y = 'loan_amount', data = data, ax=ax[0,0]).set_title('Movable assets vs loan_amount')
sns.scatterplot(x='Immovable_assets', y = 'loan_amount', data = data, ax=ax[0,1]).set_title('Immovable assets vs loan_amount')
sns.scatterplot(x='income_annum', y = 'loan_amount', data = data, ax=ax[1,0]).set_title('Income per annum vs laon amount')
sns.boxplot(x='no_of_dependents', y='loan_amount', data = data, ax=ax[1,1]).set_title('No of dependents vs loan amount')

for axes_row in ax:
    for axes in axes_row:
        axes.set_xticklabels(axes.get_xticklabels(), rotation=0, ha='right')
        axes.set_title(axes.get_title(), fontsize=12)


plt.tight_layout()
plt.show()

# %%
data.info()

# %% [markdown]
# **Data Preprocessing**

# %%

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data["education"] = label_encoder.fit_transform(data["education"])
data["self_employed"] = label_encoder.fit_transform(data["self_employed"])
data["loan_status"] = label_encoder.fit_transform(data["loan_status"]) 

data.head(2)
            

# %% [markdown]
# correlation heatmap

# %%
plt.figure(figsize=(8,5))
sns.heatmap(data.corr(), annot=True, cmap='inferno')

# %% [markdown]
# Train-test split

# %%
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(data.drop('loan_status', axis=1), data['loan_status'], test_size=0.2, random_state=42)

print('training set:', X_train.shape, y_train.shape)
print('Testing set:', X_test.shape, y_test.shape)

# %% [markdown]
# Model development for loan approval prediction.
# 
# - I'll be using the following selected machine learning algorithms to predict the loan approval status.
# 1. Random forest classifier
# 2. Xgboost classifier
# 3. Catboost classifier
# 
# combining the GridSearch and BayesSearch for the hyperparameter tuning as to get the best parameter.

# %%
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# %% [markdown]
# **Random Forest Classifier**

# %%
rfc_model = RandomForestClassifier()

rfc_params = {
    'n_estimators':[100,200,300],
    'max_depth':[None,10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'max_features':['auto','sqrt','log2'],
    'bootstrap':[True, False]
}

Bayes_search = BayesSearchCV(rfc_model, rfc_params, cv=5, error_score='raise')
Bayes_search.fit(X_train, y_train)

print('Best params:', Bayes_search.best_params_)
print('Best score (accuracy):', Bayes_search.best_score_)

best_rfc_params = Bayes_search.best_params_

# %%
rfc_model = RandomForestClassifier(**best_rfc_params)
rfc_model.fit(X_train, y_train)

rfc_model

# %%
rfc_pred = rfc_model.predict(X_test)

rfc_pred[0:5]

# %% [markdown]
# **XGBoost Classifier**

# %%
xgb_model = XGBClassifier()

xgb_params = {
    'n_estimators':[100,200,300],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1,0.2],
    'subsample':[0.8,0.9,1.0],
    'colsample_bytree':[0.8,0.9,1.0],
    'gamma':[0,0.1,0.2],
    'min_child_weight':[1,2,3]
}

Grid_search = GridSearchCV(xgb_model, xgb_params, cv=5, error_score='raise')
Grid_search.fit(X_train, y_train)

print('Best parameter:', Grid_search.best_params_)
print('Best Score(Accuracy):', Grid_search.best_score_)

Best_xgb_params = Grid_search.best_params_

# %%
xgb_model = XGBClassifier(**Best_xgb_params)
xgb_model.fit(X_train, y_train)

xgb_model

# %%
xgb_pred = xgb_model.predict(X_test)

xgb_pred[0:5]

# %% [markdown]
# **CatBoost Classifier**

# %%
cat_model = CatBoostClassifier()

cat_params = {
    'learning_rate':[0.01,0.1,0.2],
    'depth':[1,5,7],
    'l2_leaf_reg':[0,0.1,0.2],
    'subsample':[0.1,0.2,0.3],
    
    }

Grid_search = GridSearchCV(cat_model, cat_params, cv=5, error_score='raise')
Grid_search.fit(X_train, y_train)

print('Best parameter:', Grid_search.best_params_)
print('Best Score(Accuracy):', Grid_search.best_score_)

Best_cat_params = Grid_search.best_params_

# %%
cat_model = CatBoostClassifier(**Best_cat_params)
cat_model.fit(X_train, y_train)

cat_model

# %%
cat_pred = cat_model.predict(X_test)

cat_pred[0:5]

# %% [markdown]
# **Model Evaluation**
# 
# - Confusion Matrix

# %%
def Plot_confusionmatrix(y_true, y_preds, model_names):
    fig, axes = plt.subplots(1, len(y_preds), figsize=(15,6))
    
    for i, (pred, model_name) in enumerate(zip(y_preds, model_names)):
        
        cm = confusion_matrix(y_true, pred)
        ax = axes[i]
        sns.heatmap(cm, annot=True, ax=ax, cmap='viridis')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.xaxis.set_ticklabels(['False Positive', 'True Negative'])
        ax.yaxis.set_ticklabels(['True Positive ', 'False Negative'])
    
    plt.tight_layout()
    plt.show()

# %%
y_true = y_test
y_preds = [rfc_pred, xgb_pred, cat_pred]
model_names = ['Random Forest Classifier','XGBoost Classifier','CatBoost Classifier']

Plot_confusionmatrix(y_true, y_preds, model_names)

# %% [markdown]
# - True Positive (TP): The model correctly predicts positive instances as positive.
# - False Positive (FP): The model incorrectly predicts negative instances as positive.
# - False Negative (FN): The model incorrectly predicts positive instances as negative.
# - True Negative (TN): The model correctly predicts negative instances as negative.
# 
# The confusion matrix heatmap visualizes the true positive and true negative value counts in the 3 machine learning models.
# `The Random Forest classifier` has only `18` false positive and negative values.\
# `The XGBoost classifier` has `10` false positive and negative values.\
# `The CatBoost classifier` has `21` false positive and negative values.\
# 
# overall, the `XGBoost classifier` has a better accuracy compared to other listed above.

# %% [markdown]
# **Performance report**

# %%
r2_score_xgb = r2_score(y_test, xgb_pred)
mse_xgb = mean_squared_error(y_test, xgb_pred)
mae_xgb = mean_absolute_error(y_test, xgb_pred)


r2_score_rf = r2_score(y_test, rfc_pred)
mse_rf = mean_squared_error(y_test, rfc_pred)
mae_rf = mean_absolute_error(y_test, rfc_pred)


r2_score_cat = precision_score(y_test, cat_pred)
mse_cat = mean_squared_error(y_test, cat_pred)
mae_cat = mean_absolute_error(y_test, cat_pred)


# %%

list_r2_score = [r2_score_rf, r2_score_xgb, r2_score_cat]
list_mse = [mse_rf, mse_xgb, mse_cat]
list_mae = [mae_rf, mae_xgb, mae_cat]


Report = pd.DataFrame(list_r2_score, index=['RandomForest Classifier','XGBoost Classifier','CatBoost Classifier'])
Report.columns =['R2 Score']
Report.insert(loc=1,column='Mean Squared Error',value=list_mse)
Report.insert(loc=2, column='Mean Absolute Error', value=list_mae)

Report.columns.name = 'Algorithm'
print(Report)

# %% [markdown]
# **Classification Report**

# %%
print(f"XGBoostClassifier \n\n{classification_report(y_test, xgb_pred)}")
print(f"CatBoostClassifier \n\n{classification_report(y_test, cat_pred)}")
print(f"RandomForestClassifier \n\n{classification_report(y_test, rfc_pred)}")


# %% [markdown]
# **Conclusion**
# 
# Based on the metrics, visualizations, and analysis provided, the `XGBoost Classifier` emerges as the preferred machine learning model for predicting loan approval status. This conclusion is drawn from its higher accuracy of `99%`, compared to `98%` for both the `Random Forest Classifier` and the `CatBoost Classifier`. Consequently, the `XGBoost Classifier` outperformed the other models, demonstrating superior predictive capabilities in this scenario.


