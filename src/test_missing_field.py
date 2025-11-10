import requests

url = "http://127.0.0.1:8000/predict"

# Missing "weight" and other fields
data = {"age": 55, "height": 172}

try:
    response = requests.post(url, json=data)
    print("Status code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
# This test checks how the API handles a request with missing required fields.
