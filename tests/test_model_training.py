import pytest
from src.model_pipeline import train_model, train_decision_tree, get_hyperparameters

import pytest
from sklearn.preprocessing import LabelEncoder
from src.model_pipeline import train_model, get_hyperparameters

def test_train_model():
    """Test that the model is trained correctly."""
    # Mock data
    X_train = [
        [1, 2, 3, "Yes", 10],
        [4, 5, 6, "No", 20],
        [7, 8, 9, "Yes", 30],
        [10, 11, 12, "No", 40],
        [13, 14, 15, "Yes", 50],
        [16, 17, 18, "No", 60],
        [17, 18, 19, "Yes", 70],
        [20, 21, 22, "No", 80],
        [23, 24, 25, "Yes", 90],
        [26, 27, 28, "No", 100]
    ]
    y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    # Encode categorical features
    label_encoder = LabelEncoder()
    for i in range(len(X_train)):
        X_train[i][3] = label_encoder.fit_transform([X_train[i][3]])[0]  # Encode "Yes"/"No" to 1/0

    # Train the model
    hyperparameters = get_hyperparameters()
    model = train_model(X_train, y_train, hyperparameters)

    # Assert that the model is trained
    assert model is not None

def test_train_decision_tree():
    """Test that the decision tree model is trained correctly."""
    # Mock data
    X_train = [
        [1, 2, 3, "Yes", 10],
        [4, 5, 6, "No", 20],
        [7, 8, 9, "Yes", 30],
        [10, 11, 12, "No", 40],
        [13, 14, 15, "Yes", 50],
        [16, 17, 18, "No", 60],
        [17, 18, 19, "Yes", 70],
        [20, 21, 22, "No", 80],
        [23, 24, 25, "Yes", 90],
        [26, 27, 28, "No", 100]
    ]
    y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] 
    # Encode categorical features
    label_encoder = LabelEncoder()
    for i in range(len(X_train)):
        X_train[i][3] = label_encoder.fit_transform([X_train[i][3]])[0]  # Encode "Yes"/"No" to 1/0

    # Train the model
    model = train_decision_tree(X_train, y_train)

    # Assert that the model is trained
    assert model is not None