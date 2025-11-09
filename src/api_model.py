# api_model.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

import os
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# --- Initialize the API ---
app = FastAPI(
    title="Diabetes Prediction API",
    description="API that serves a trained ML model for predicting diabetes mellitus",
    version="1.0"
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
        # Convert input to array (reshape for model)
        features = np.array([[data.age, data.height, data.weight, data.aids, data.cirrhosis,
                              data.hepatic_failure, data.immunosuppression, data.leukemia,
                              data.lymphoma, data.solid_tumor_with_metastasis]])

        # Make prediction (probability if available)
        pred = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else model.predict(features)[0]

        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
