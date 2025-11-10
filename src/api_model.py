# api_model.py
import logging  # For logging errors and information
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
except Exception as e:
    raise Exception(f"Error loading model: {e}")


# --- Initialize the API ---
# Creating a FastAPI application where the title/description appear in the auto-generated documentation
app = FastAPI(
    title="Diabetes Prediction API",
    description="API that serves a trained ML model for predicting diabetes mellitus",
    version="1.0",
)


# --- Define the expected input schema ---
# BaseModel from Pydantic does automatic validation
# e.g.: If someone sends age: "fifty", FastAPI will reject it (not a float!)
# e.g.: If someone forgets weight, FastAPI will say "missing required field"
class PatientData(BaseModel):
    age: float  # age in years
    height: float  # height in cm
    weight: float  # weight in kg
    aids: int  # 1 if the patient has AIDS, 0 otherwise
    cirrhosis: int  # 1 if the patient has cirrhosis, 0 otherwise
    hepatic_failure: int  # 1 if the patient has hepatic failure, 0 otherwise
    immunosuppression: int  # 1 if the patient is immunosuppressed, 0 otherwise
    leukemia: int  # 1 if the patient has leukemia, 0 otherwise
    lymphoma: int  # 1 if the patient has lymphoma, 0 otherwise
    solid_tumor_with_metastasis: (
        int  # 1 if the patient has solid tumor with metastasis, 0 otherwise
    )


# --- Define the prediction endpoint ---
# @app.post("/predict") is a decorator that tells FastAPI: "When someone sends a POST request to /predict, run this function"
@app.post("/predict")
def predict(data: PatientData):  # function receives data of type PatientData
    try:
        # Convert input to a one-row DataFrame
        data_dict = data.model_dump()
        # data.model_dump() converts the Pydantic model to a regular Python dictionary
        # (older Pydantic versions used data.dict() instead)
        df = pd.DataFrame([data_dict])
        # creating one-row DataFrame, because scikit-learn models expect DataFrames/arrays as input
        # [data_dict] wraps it in a list to make it a single row

        # If the model has feature_names_in_, align to that order
        if hasattr(model, "feature_names_in_"):
            # hasattr checks if the model has the attribute feature_names_in_
            # This exists for scikit-learn models trained on DataFrames (stores column names)
            # Reorders the DataFrame columns to match the training order
            # e.g. model trained on ['weight', 'age', 'height'] but input is ['age', 'height', 'weight'], gets fixed
            df = df[list(model.feature_names_in_)]

        # Make prediction (probability if available)
        if hasattr(model, "predict_proba"):  # probabilities for classification models
            proba = model.predict_proba(df)[0]  # full probability array
            pred_class = int(
                proba[1] >= 0.5
            )  # Classify as 1 if prob of positive class >= 0.5 else 0
            return {
                "prediction": float(
                    pred_class
                ),  # return as float for JSON serialization
                "probability": float(
                    proba[1]
                ),  # return as float for JSON serialization
                "confidence": float(
                    max(proba)
                ),  # return as float for JSON serialization
            }
        else:
            pred = model.predict(df)[0]
            return {"prediction": float(pred)}  # return as float for JSON serialization

    except Exception as e:
        logging.error(f"Prediction error: {e}")  # Log the error for debugging
        raise HTTPException(status_code=500, detail=str(e))
