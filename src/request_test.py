# request_test.py
import json
import os

import requests

# define where the API is running
url = "http://127.0.0.1:8000/predict"

# load the test data from example_input.json wherever this script is located
file_path = os.path.join(os.path.dirname(__file__), "example_input.json")
with open(file_path) as f:
    data = json.load(f)

# send HTTP POST request to the API
try:
    response = requests.post(url, json=data)
    # 'requests` is a Python library for making HTTP requests
    # `post()` sends a **POST request** (remember: POST = send data to server)
    # `json=data` converts the Python dictionary to JSON and sends it as the request body
    print("Status code:", response.status_code)
    print("Response body:", response.text)  # <- shows {"detail": "..."} on error
    response.raise_for_status()
    print("✅ Prediction from API:", response.json())
except requests.exceptions.RequestException as e:
    print("❌ API Request failed:", e)

# Diagram of the flow:
"""
[request_test.py] 
    ↓ (HTTP POST request with JSON data)
[http://127.0.0.1:8000/predict]
    ↓ (FastAPI receives it)
[api_model.py - predict() function]
    ↓ (Validates with Pydantic)
[PatientData model]
    ↓ (Converts to DataFrame)
[model.pkl - your trained ML model]
    ↓ (Makes prediction)
[Returns JSON response]
    ↓
[request_test.py receives response]
"""
# Note: The FastAPI server needs to be running before executing this script!
# start it with: uvicorn api_model:app --reload

# ensure that example_input.json exists in the same directory as this script
# with valid test data matching the PatientData schema.
