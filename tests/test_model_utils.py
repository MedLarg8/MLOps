import pytest
import joblib
from sklearn.tree import DecisionTreeClassifier
from src.model_pipeline import save_model, load_model

def test_save_and_load_model(tmpdir):
    """Test that a model can be saved and loaded correctly."""
    from pathlib import Path
    from sklearn.tree import DecisionTreeClassifier
    from src.model_pipeline import save_model, load_model
    import joblib

    # Create a dummy model
    model = DecisionTreeClassifier()

    # Convert tmpdir to a string path
    model_path = Path(str(tmpdir.join("test_model.joblib")))

    # Save the model
    save_model(model, model_path)

    # Load the model
    loaded_model = load_model(model_path)

    # Assert model is loaded correctly
    assert isinstance(loaded_model, DecisionTreeClassifier)
