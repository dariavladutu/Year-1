# Basic Libraries
import os
import time
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

# Database Libraries
import psycopg2
import sqlite3
from sqlalchemy import create_engine, text, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Data Preprocessing and Feature Engineering
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Model Selection and Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score, top_k_accuracy_score

# Machine Learning Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Advanced Models
from xgboost import XGBClassifier

# Oversampling
from imblearn.over_sampling import SMOTE

# Interpretability
import lime
import lime.lime_tabular

# Statistical Libraries
import statsmodels.api as sm

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# Set Plot Style
plt.style.use('Solarize_Light2')

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled dataset
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)

rf_res = RandomForestClassifier(random_state=42)

# Train the RandomForest model on the resampled dataset
rf_res.fit(X_train_res, y_train_res)

# Predict and evaluate
y_pred_test_res_rf = rf_res.predict(X_test_res)
print("RandomForest Test Accuracy for Risk Level Classification:", accuracy_score(y_test_res, y_pred_test_res_rf))
print(classification_report(y_test_res, y_pred_test_res_rf))

# Plot confusion matrix
plot_confusion_matrix(y_test_res, y_pred_test_res_rf, 'RandomForest Balanced Dataset Confusion Matrix')

# Cross Validation
rf_res = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf_res, X_train_res, y_train_res, cv=5) # 5-fold cross-validation

print("Accuracy for each fold for 5-fold cross-validation:", np.mean(np.abs(cv_scores))) # accuracy for each fold
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Measure training time
start_time = time.time()

# Train the RandomForest model
rf_res.fit(X_train_res, y_train_res)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Get feature importance
importances = rf_res.feature_importances_
feature_names = X_train_res.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# Ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="lime")

# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_res.values,
    feature_names=X_train_res.columns,
    class_names=y_train_res.unique(),
    mode='classification'
)

# Define a function that accepts a data frame
def predict_fn(x):
    x_df = pd.DataFrame(x, columns=X_train_res.columns)
    return rf_res.predict_proba(x_df)

# Explain a prediction
i = 0                            # index of the instance to explain
exp = explainer.explain_instance(
    data_row=X_test_res.iloc[i],
    predict_fn=predict_fn
)

# Show explanation
exp.show_in_notebook(show_all=False)

# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_res.values,
    feature_names=X_train_res.columns,
    class_names=y_train_res.unique(),
    mode='classification'
)

# Define a function that accepts a data frame
def predict_fn(x):
    x_df = pd.DataFrame(x, columns=X_train_res.columns)
    return rf_res.predict_proba(x_df)

# Explain a prediction
i = 0                            # index of the instance to explain
exp = explainer.explain_instance(
    data_row=X_test_res.iloc[i],
    predict_fn=predict_fn
)

# Show explanation
exp.show_in_notebook(show_all=False)

# Explain a prediction
i = 1  # index of the instance to explain
exp = explainer.explain_instance(
    data_row=X_test_res.iloc[i],
    predict_fn=predict_fn
)

# Show explanation
exp.show_in_notebook(show_all=False)

# Explain a prediction
i = 1500 # index of the instance to explain
exp = explainer.explain_instance(
    data_row=X_test_res.iloc[i],
    predict_fn=predict_fn
)

# Show explanation
exp.show_in_notebook(show_all=False)