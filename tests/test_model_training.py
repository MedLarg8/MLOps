import pytest
from src.model_pipeline import train_model, train_decision_tree, get_hyperparameters

def test_train_model():
    """Test that the model is trained correctly."""
    # Mock data
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = [0, 1, 0]

    # Train the model
    hyperparameters = get_hyperparameters()
    model = train_model(X_train, y_train, hyperparameters)

    # Assert that the model is trained
    assert model is not None

def test_train_decision_tree():
    """Test that the decision tree model is trained correctly."""
    # Mock data
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = [0, 1, 0]

    # Train the model
    model = train_decision_tree(X_train, y_train)

    # Assert that the model is trained
    assert model is not None