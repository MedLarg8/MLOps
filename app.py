from fastapi import FastAPI, Request, Form, HTTPException  # Import HTTPException here
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import requests
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Load the trained model
model = joblib.load("decision_tree_model.joblib")

# Set up templates for HTML
templates = Jinja2Templates(directory="templates")

class InputData(BaseModel):
    entree: list

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

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, input_data: str = Form(...)):
    numbers = list(map(float, input_data.split(",")))
    response = requests.post("http://localhost:8000/predict", json={"entree": numbers})
    prediction = response.json().get("prediction", "Error in prediction")
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)