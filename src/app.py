# app.py to create application
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import logging
from pathlib import Path

# This line will create the 'logs' directory if it doesn't already exist.
Path("logs").mkdir(exist_ok=True)

# Setup basic logging
logging.basicConfig(level=logging.INFO, filename="logs/predictions.log", format='%(asctime)s - %(message)s')

# Define the input data schema using Pydantic (Bonus) [cite: 18]
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

app = FastAPI()

# Load the production model from the MLflow Model Registry
# Find the correct Experiment ID (the folder name inside mlruns)
EXPERIMENT_ID = "375657399447121571"
# Paste the Run ID you copied from the MLflow UI
RUN_ID = "m-a5eec3f715b64828a7a9fab5ab5e5a48" 

# Construct the correct path to the model artifact inside the container
logged_model_uri = f"/app/mlruns/{EXPERIMENT_ID}/models/{RUN_ID}/artifacts/."

# This line stays the same
model = mlflow.pyfunc.load_model(logged_model_uri)
#model = mlflow.pyfunc.load_model(logged_model)

@app.post("/predict")
def predict(features: HouseFeatures):
    """Accepts housing features and returns a prediction."""
    # Convert Pydantic model to a DataFrame for prediction
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)
    result = prediction[0]

    # Log the input and output 
    logging.info(f"Input: {features.dict()} | Prediction: {result}")

    # Return the prediction in JSON format [cite: 12]
    return {"predicted_median_house_value": result}

@app.get("/")
def read_root():
    return {"status": "ok"}