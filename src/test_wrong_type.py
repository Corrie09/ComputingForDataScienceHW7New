import requests

url = "http://127.0.0.1:8000/predict"

# "age" is a string instead of float
data = {
    "age": "fifty-five",  # ❌ Should be a number!
    "height": 172,
    "weight": 78,
    "aids": 0,
    "cirrhosis": 0,
    "hepatic_failure": 0,
    "immunosuppression": 0,
    "leukemia": 0,
    "lymphoma": 0,
    "solid_tumor_with_metastasis": 1,
}

try:
    response = requests.post(url, json=data)
    print("Status code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
# This test checks how the API handles a request with incorrect data types.
