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
import logging

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

# TensorFlow and Keras for Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Plot Style
plt.style.use('Solarize_Light2')

def main():
    """
    Main function to apply SMOTE, train and evaluate Decision Tree, Random Forest, and DNN models,
    and interpret results using LIME.
    """
    try:
        

        logging.info("Applying SMOTE to balance the dataset.")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        logging.info("Splitting the resampled dataset.")
        X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)

        # Decision Tree Classifier
        logging.info("Training the Decision Tree model on the resampled dataset.")
        dt_clf = DecisionTreeClassifier(random_state=42)
        dt_clf.fit(X_train_res, y_train_res)

        logging.info("Predicting and evaluating the Decision Tree model.")
        y_pred_test_res_dt = dt_clf.predict(X_test_res)
        print("Decision Tree Test Accuracy for Risk Level Classification:", accuracy_score(y_test_res, y_pred_test_res_dt))
        print(classification_report(y_test_res, y_pred_test_res_dt))

        logging.info("Plotting confusion matrix for Decision Tree.")
        plot_confusion_matrix(y_test_res, y_pred_test_res_dt, 'Decision Tree Balanced Dataset Confusion Matrix')

        # Random Forest Classifier
        logging.info("Training the Random Forest model on the resampled dataset.")
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(X_train_res, y_train_res)

        logging.info("Predicting and evaluating the Random Forest model.")
        y_pred_test_res_rf = rf_clf.predict(X_test_res)
        print("Random Forest Test Accuracy for Risk Level Classification:", accuracy_score(y_test_res, y_pred_test_res_rf))
        print(classification_report(y_test_res, y_pred_test_res_rf))

        logging.info("Plotting confusion matrix for Random Forest.")
        plot_confusion_matrix(y_test_res, y_pred_test_res_rf, 'Random Forest Balanced Dataset Confusion Matrix')

        logging.info("Performing cross-validation for Random Forest.")
        cv_scores = cross_val_score(rf_clf, X_train_res, y_train_res, cv=5)
        print("Accuracy for each fold for 5-fold cross-validation:", np.mean(np.abs(cv_scores)))
        print("Cross-Validation Scores:", cv_scores)
        print("Mean Cross-Validation Score:", cv_scores.mean())

        logging.info("Measuring training time for Random Forest.")
        start_time = time.time()
        rf_clf.fit(X_train_res, y_train_res)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")

        logging.info("Getting feature importance for Random Forest.")
        importances = rf_clf.feature_importances_
        feature_names = X_train_res.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        logging.info("Plotting feature importance for Random Forest.")
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.show()

        logging.info("Creating LIME explainer for Random Forest.")
        warnings.filterwarnings("ignore", category=FutureWarning, module="lime")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_res.values,
            feature_names=X_train_res.columns,
            class_names=y_train_res.unique(),
            mode='classification'
        )

        def predict_fn(x):
            """
            Prediction function for LIME explainer.
            """
            x_df = pd.DataFrame(x, columns=X_train_res.columns)
            return rf_clf.predict_proba(x_df)

        for i in [0, 1, 1500]:  # indices of the instances to explain
            logging.info(f"Explaining prediction for instance {i}.")
            exp = explainer.explain_instance(
                data_row=X_test_res.iloc[i],
                predict_fn=predict_fn
            )
            exp.show_in_notebook(show_all=False)

        # Deep Neural Network Model
        logging.info("Building and training the Deep Neural Network (DNN) model.")
        train_dnn_model(X_resampled, y_resampled)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plot confusion matrix using seaborn.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    title (str): Title for the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap="YlGnBu", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.ylabel('Actual values')
    plt.xlabel('Predicted values')
    plt.title(title)
    plt.show()

def train_dnn_model(X, y):
    """
    Builds, trains, and evaluates a Deep Neural Network (DNN) model.

    Parameters:
    X (pd.DataFrame): The preprocessed dataframe with features.
    y (pd.Series): The target variable.
    """
    try:
        logging.info("Selecting relevant features and target variable.")
        y_encoded = to_categorical(y)

        logging.info("Splitting the dataset into training and testing sets.")
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        logging.info("Standardizing the features.")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        logging.info("Building the DNN model.")
        model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dense(y_encoded.shape[1], activation='softmax')
        ])

        logging.info("Compiling the model.")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        logging.info("Defining EarlyStopping callback.")
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        logging.info("Training the model.")
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        logging.info("Evaluating the model.")
        loss, accuracy = model.evaluate(X_test, y_test)
        logging.info(f'Loss: {loss}, Accuracy: {accuracy}')

        logging.info("Making predictions.")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        logging.info("Calculating evaluation metrics.")
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        class_report = classification_report(y_test_classes, y_pred_classes)

        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        logging.info(f"Classification report:\n{class_report}")

        logging.info("Plotting the confusion matrix.")
        sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt='g', linewidths=2)
        plt.ylabel('Actual values')
        plt.xlabel('Predicted values')
        plt.show()

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()