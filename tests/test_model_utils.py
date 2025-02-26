import pytest
import joblib
from sklearn.tree import DecisionTreeClassifier
from src.model_pipeline import save_model, load_model

def test_save_and_load_model(tmpdir):
    """Test that a model can be saved and loaded correctly."""
    # Create a dummy model
    model = DecisionTreeClassifier()

    # Save the model
    model_path = tmpdir.join("test_model.joblib")
    save_model(model, model_path)

    # Load the model
    loaded_model = load_model(model_path)

    # Assert that the loaded model is the same as the saved model
    assert isinstance(loaded_model, DecisionTreeClassifier)