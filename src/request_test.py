# request_test.py
import requests
import json
import os

url = "http://127.0.0.1:8000/predict"

file_path = os.path.join(os.path.dirname(__file__), "example_input.json")
with open(file_path) as f:
    data = json.load(f)

try:
    response = requests.post(url, json=data)
    print("Status code:", response.status_code)
    print("Response body:", response.text)  # <- shows {"detail": "..."} on error
    response.raise_for_status()
    print("✅ Prediction from API:", response.json())
except requests.exceptions.RequestException as e:
    print("❌ API Request failed:", e)
