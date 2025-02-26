"""
This module contains functions and processes related to the model
pipeline for data analysis.

It includes steps for data preprocessing, feature selection, and
model training.
The focus is on processing the dataset, selecting relevant features,
and preparing the data
for machine learning.

Functions:
- Data preprocessing: Cleans and prepares the dataset for modeling.
- Feature selection: Chooses the most important features for
training the model.
- Model training: Trains a machine learning model on the processed data.

Usage:
1. Load the dataset into a pandas DataFrame.
2. Apply preprocessing and feature selection functions.
3. Train a model and evaluate its performance.

Author: 3ami chat
Date: 26/02/2025
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


def load_data(file_path):
    """Load the dataset from a file."""
    return pd.read_csv(file_path)


def engineer_features(data):
    """Perform feature engineering."""
    data["Total minutes"] = (
        data["Total day minutes"]
        + data["Total eve minutes"]
        + data["Total night minutes"]
        + data["Total intl minutes"]
    )
    data["Total charge"] = (
        data["Total day charge"]
        + data["Total eve charge"]
        + data["Total night charge"]
        + data["Total intl charge"]
    )
    data["Total calls"] = (
        data["Total day calls"]
        + data["Total eve calls"]
        + data["Total night calls"]
        + data["Total intl calls"]
    )
    return data


def preprocess_data(data):
    """Preprocess the data (encoding, scaling, etc.)."""
    selected_features = [
        "Total minutes",
        "Total charge",
        "Total calls",
        "International plan",
        "Customer service calls",
    ]
    x = data[selected_features]
    y = data["Churn"]
    # Handle categorical 'International plan' column
    label_encoder = LabelEncoder()
    int_plan = "International plan"
    x.loc[:, int_plan] = label_encoder.fit_transform(x[int_plan])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y, scaler, label_encoder


def split_data(x, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(x, y, test_size=test_size,
                            random_state=random_state)


def prepare_data(file_path):
    """Load, engineer features, preprocess, and split the data."""
    print("Loading data...")
    data = load_data(file_path)
    print("Engineering features...")
    data = engineer_features(data)
    print("Preprocessing data...")
    x, y, scaler, label_encoder = preprocess_data(data)
    print("Splitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = split_data(x, y)
    return x_train, x_test, y_train, y_test, scaler, label_encoder


def get_hyperparameters():
    """Return a dictionary of hyperparameters for GridSearchCV."""
    return {
        "criterion": ["gini"],
        "max_depth": [None],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": [None],
        "splitter": ["best"],
        "max_leaf_nodes": [20],
    }


def train_model(x_train, y_train, hyperparameters, random_state=42, cv=5):
    """Train a model using GridSearchCV."""
    print("Starting model training...")
    model = DecisionTreeClassifier(random_state=random_state)
    print("Model initialized.")
    # Initialize GridSearchCV
    print("Initializing GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=model, param_grid=hyperparameters, cv=cv, n_jobs=-1,
        verbose=1
    )
    print("Starting GridSearchCV fitting...")
    grid_search.fit(x_train, y_train)
    print("GridSearchCV fitting completed.")
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_


def train_decision_tree(x_train, y_train):
    """Train a Decision Tree model."""
    hyperparameters = get_hyperparameters()
    return train_model(x_train, y_train, hyperparameters)


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    return accuracy, conf_matrix, class_report


def print_metrics(accuracy, conf_matrix, class_report):
    """Print evaluation metrics."""
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")


def evaluate_model(model, x_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(x_test)
    accuracy, conf_matrix, class_report = calculate_metrics(y_test, y_pred)
    print_metrics(accuracy, conf_matrix, class_report)
    return accuracy, conf_matrix, class_report


def plot_confusion_matrix(conf_matrix):
    """
    Plot a heatmap of the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churn", "Churn"],
        yticklabels=["Not Churn", "Churn"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Decision Tree")
    plt.show()


def save_model(model, filename):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Load a trained model from a file.
    """
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
