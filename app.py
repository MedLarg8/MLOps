from fastapi import FastAPI, Request, Form, HTTPException  # Import HTTPException here
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import requests
from fastapi.templating import Jinja2Templates
from sklearn.tree import DecisionTreeClassifier
from src.model_pipeline import load_data, engineer_features, preprocess_data, split_data
from sklearn.model_selection import GridSearchCV
from typing import Optional
from pydantic import BaseModel, ValidationError, validator

app = FastAPI()

# Load the trained model
model = joblib.load("decision_tree_model.joblib")

# Set up templates for HTML
templates = Jinja2Templates(directory="templates")

class InputData(BaseModel):
    entree: list

class RetrainModelRequest(BaseModel):
    criterion: str
    max_depth: Optional[int] = None  # Allow null values
    min_samples_split: int
    min_samples_leaf: int
    max_features: Optional[str] = None  # Allow null values
    splitter: str
    max_leaf_nodes: int

    # Custom validator for max_depth
    @validator('max_depth')
    def validate_max_depth(cls, value):
        if value == 0:  # Treat 0 as None
            return None
        if value is not None and value < 0:
            raise ValueError("max_depth must be a positive integer or 0 (for None).")
        return value

    # Custom validator for max_features
    @validator('max_features')
    def validate_max_features(cls, value):
        if value == "0":  # Treat "0" as None
            return None
        if value is not None and value.lower() not in ["sqrt", "log2"] and not value.replace(".", "", 1).isdigit():
            raise ValueError("max_features must be 'sqrt', 'log2', a number, or 0 (for None).")
        return value

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Ensure the input is a list of numbers and that it's not empty
        if not isinstance(data.entree, list) or not all(isinstance(i, (int, float)) for i in data.entree):
            raise HTTPException(status_code=400, detail="Input data must be a list of numbers.")

        # Ensure that the model can predict on this data (e.g., correct number of features)
        if len(data.entree) != model.n_features_in_:
            raise HTTPException(status_code=400, detail=f"Input should have {model.n_features_in_} features.")
        
        prediction = model.predict([data.entree])
        return {"prediction": prediction.tolist()}
    
    except HTTPException as e:
        # Return the error message directly
        return {"error": e.detail}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Serve HTML page on another port (e.g., 5000)
@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.get("/retrain", response_class=HTMLResponse)
async def retrain_form(request: Request):
    return templates.TemplateResponse("retrain.html", {"request": request})

# Handle the retrain form submission
@app.post("/retrain")
async def retrain_model(request: RetrainModelRequest):
    try:
        # Load and preprocess data
        raw_data = load_data("merged_churn.csv")
        engineered_data = engineer_features(raw_data)
        X, y, scaler, label_encoder = preprocess_data(engineered_data)

        # Convert max_features to the correct type
        if request.max_features and request.max_features.lower() == "none":
            max_features = None
        elif request.max_features and request.max_features.isdigit():
            max_features = int(request.max_features)
        elif request.max_features and request.max_features.replace(".", "", 1).isdigit():  # Check if it's a float
            max_features = float(request.max_features)
        elif request.max_features and request.max_features.lower() in ["sqrt", "log2"]:
            max_features = request.max_features.lower()
        else:
            raise ValueError("Invalid value for max_features. Must be 'sqrt', 'log2', a number, or 0 (for None).")

        # Define hyperparameters from the request data
        hyperparameters = {
            "criterion": [request.criterion],
            "max_depth": [request.max_depth],  # This can be None
            "min_samples_split": [request.min_samples_split],
            "min_samples_leaf": [request.min_samples_leaf],
            "max_features": [max_features],  # This can be None
            "splitter": [request.splitter],
            "max_leaf_nodes": [request.max_leaf_nodes],
        }

        # Train the model using GridSearchCV
        model = DecisionTreeClassifier()
        grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X, y)

        # Save the updated model
        joblib.dump(grid_search.best_estimator_, "decision_tree_model4.joblib")

        # Return a success message
        return {"message": "Model retrained successfully!", "best_params": grid_search.best_params_}
    
    except ValidationError as e:
        # Handle Pydantic validation errors
        errors = []
        for error in e.errors():
            field = error["loc"][0]
            msg = error["msg"]
            errors.append(f"{field}: {msg}")
        raise HTTPException(status_code=422, detail={"errors": errors})
    except Exception as e:
        # Handle other exceptions
        raise HTTPException(status_code=400, detail=f"Retraining failed: {str(e)}")

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, input_data: str = Form(...)):
    numbers = list(map(float, input_data.split(",")))
    response = requests.post("http://localhost:8000/predict", json={"entree": numbers})
    prediction = response.json().get("prediction", "Error in prediction")
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)