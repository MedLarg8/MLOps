import pytest
from src.model_pipeline import load_data, engineer_features, preprocess_data, split_data

def test_load_data():
    """Test that data is loaded correctly."""
    data = load_data("merged_churn.csv")
    assert data is not None
    assert not data.empty

def test_engineer_features():
    """Test that features are engineered correctly."""
    data = load_data("merged_churn.csv")
    engineered_data = engineer_features(data)
    assert "Total minutes" in engineered_data.columns
    assert "Total charge" in engineered_data.columns
    assert "Total calls" in engineered_data.columns

def test_preprocess_data():
    """Test that data is preprocessed correctly."""
    data = load_data("merged_churn.csv")
    engineered_data = engineer_features(data)
    X, y, scaler, label_encoder = preprocess_data(engineered_data)
    assert X is not None
    assert y is not None
    assert scaler is not None
    assert label_encoder is not None

def test_split_data():
    """Test that data is split correctly."""
    data = load_data("merged_churn.csv")
    engineered_data = engineer_features(data)
    X, y, scaler, label_encoder = preprocess_data(engineered_data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0