# request_test.py
import requests
import json

url = "http://127.0.0.1:8000/predict"

import os
file_path = os.path.join(os.path.dirname(__file__), "example_input.json")
with open(file_path) as f:
    data = json.load(f)

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raises HTTPError for bad responses
    print("✅ Prediction from API:", response.json())
except requests.exceptions.RequestException as e:
    print("❌ API Request failed:", e)
