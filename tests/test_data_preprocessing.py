import pytest
from model_pipeline import train_model, get_hyperparameters

def test_train_model():
    hyperparameters = get_hyperparameters()
    X_train = [[1, 2], [3, 4]]
    y_train = [0, 1]
    model = train_model(X_train, y_train, hyperparameters)
    assert model is not None