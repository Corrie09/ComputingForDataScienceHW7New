# api_model.py
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load model trained on the 10 FEATURES only
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
# gets the directory where api_model.py lives and builds the full path to model.pkl in that same directory
try:
    model = joblib.load(model_path)  # loads the trained model from the model.pkl file
except FileNotFoundError:
    raise FileNotFoundError(
        f"Model file not found at {model_path}. Please ensure the model.pkl file is in the correct location."
    )


# --- Initialize the API ---
app = FastAPI(
    title="Diabetes Prediction API",
    description="API that serves a trained ML model for predicting diabetes mellitus",
    version="1.0",
)


# --- Define the expected input schema ---
class PatientData(BaseModel):
    age: float
    height: float
    weight: float
    aids: int
    cirrhosis: int
    hepatic_failure: int
    immunosuppression: int
    leukemia: int
    lymphoma: int
    solid_tumor_with_metastasis: int


# --- Define the prediction endpoint ---
@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input to a one-row DataFrame
        data_dict = data.model_dump()  # or data.dict() if you prefer
        df = pd.DataFrame([data_dict])

        # If the model has feature_names_in_, align to that order
        if hasattr(model, "feature_names_in_"):
            df = df[list(model.feature_names_in_)]

        # Make prediction (probability if available)
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(df)[0][1]
        else:
            pred = model.predict(df)[0]

        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
