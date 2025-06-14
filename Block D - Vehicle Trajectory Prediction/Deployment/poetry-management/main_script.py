# Importing libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        model = Sequential()
        model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(y_encoded.shape[1], activation='softmax'))

        logging.info("Compiling the model.")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        logging.info("Training the model.")
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

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
    # Example of using the functions (this part can be customized or removed as needed)
    # Load a sample DataFrame
    df = pd.read_csv('df_joined.csv')

    # Preprocessing steps
    df_selected = df[['date', 'eventid', 'duration_seconds', 'road_name', 'maxwaarde',
                      'incident_severity', 'risk_category_encoded', 'weather_code', 'risk_level_wmo']].copy()

    # Encoding categorical variables
    encoder = LabelEncoder()
    df_selected['incident_severity'] = encoder.fit_transform(df_selected['incident_severity'])
    df_selected['road_name'] = encoder.fit_transform(df_selected['road_name'].astype(str))
    df_selected['risk_level_wmo'] = encoder.fit_transform(df_selected['risk_level_wmo'])

    # Extract relevant information from 'date'
    df_selected['day_of_week'] = pd.to_datetime(df_selected['date']).dt.dayofweek
    df_selected['month'] = pd.to_datetime(df_selected['date']).dt.month
    df_selected['year'] = pd.to_datetime(df_selected['date']).dt.year

    # Drop the original 'date' column
    df_selected.drop(columns=['date'], inplace=True)

    # Selecting relevant features for this model
    X = df_selected.drop(columns=['risk_category_encoded'])  # Features
    y = df_selected['risk_category_encoded']  # Target

    # Example function calls
    plot_confusion_matrix(y, y, "Example Confusion Matrix")  # Replace with actual predictions
    train_dnn_model(X, y)
