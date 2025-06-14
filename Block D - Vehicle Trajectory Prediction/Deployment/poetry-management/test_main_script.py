import sys
import os
import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from unittest.mock import patch, MagicMock
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importing the functions to test from main_script
from main_script import plot_confusion_matrix, train_dnn_model

class TestRoadSafetyAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up data for testing.
        """
        # Load the initial DataFrame from CSV
        cls.df = pd.read_csv('df_joined.csv')
        cls.df_selected = cls.df[['date', 'eventid', 'duration_seconds', 'road_name', 'maxwaarde',
                                  'incident_severity', 'risk_category_encoded', 'weather_code', 'risk_level_wmo']].copy()

        # Encoding categorical variables
        encoder = LabelEncoder()
        cls.df_selected['incident_severity'] = encoder.fit_transform(cls.df_selected['incident_severity'])
        cls.df_selected['road_name'] = encoder.fit_transform(cls.df_selected['road_name'].astype(str))
        cls.df_selected['risk_level_wmo'] = encoder.fit_transform(cls.df_selected['risk_level_wmo'])

        # Extract relevant information from 'date'
        cls.df_selected['day_of_week'] = pd.to_datetime(cls.df_selected['date']).dt.dayofweek
        cls.df_selected['month'] = pd.to_datetime(cls.df_selected['date']).dt.month
        cls.df_selected['year'] = pd.to_datetime(cls.df_selected['date']).dt.year

        # Drop the original 'date' column
        cls.df_selected.drop(columns=['date'], inplace=True)

        # Prepare the feature matrix and target vector
        cls.X = cls.df_selected.drop(columns=['risk_category_encoded'])
        cls.y = cls.df_selected['risk_category_encoded']
        smote = SMOTE(random_state=42)
        cls.X_resampled, cls.y_resampled = smote.fit_resample(cls.X, cls.y)

    def test_csv_loading_and_selection(self):
        """
        Test loading of CSV and selection/merging of data.
        """
        self.assertFalse(self.df.empty, "The dataframe should not be empty.")
        self.assertIn('road_name', self.df.columns, "Column 'road_name' should be present in the dataframe.")
        self.assertGreater(len(self.df_selected), 0, "The selected dataframe should not be empty.")

    def test_smote(self):
        """
        Test SMOTE oversampling.
        """
        self.assertEqual(len(self.X_resampled), len(self.y_resampled))
        self.assertGreater(len(self.X_resampled), len(self.X))
    
    def test_decision_tree(self):
        """
        Test Decision Tree training and prediction.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X_resampled, self.y_resampled, test_size=0.4, random_state=42)
        dt_clf = DecisionTreeClassifier(random_state=42)
        dt_clf.fit(X_train, y_train)
        y_pred = dt_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, 0.7)

    def test_random_forest(self):
        """
        Test Random Forest training and prediction.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X_resampled, self.y_resampled, test_size=0.4, random_state=42)
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, 0.7)

    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix(self, mock_show):
        """
        Test confusion matrix plotting.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X_resampled, self.y_resampled, test_size=0.4, random_state=42)
        dt_clf = DecisionTreeClassifier(random_state=42)
        dt_clf.fit(X_train, y_train)
        y_pred = dt_clf.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, 'Test Confusion Matrix')
        mock_show.assert_called_once()

    @patch('tensorflow.keras.models.Sequential.fit')
    def test_train_dnn_model(self, mock_fit):
        """
        Test DNN model training.
        """
        mock_fit.return_value = MagicMock(history={'accuracy': [0.8], 'val_accuracy': [0.75]})
        train_dnn_model(self.X_resampled, self.y_resampled)
        mock_fit.assert_called_once()

if __name__ == '__main__':
    unittest.main()
