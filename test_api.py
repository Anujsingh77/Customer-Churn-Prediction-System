import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "age": 35,
    "balance": 60000,
    "tenure": 3
}

response = requests.post(url, json=data)
print(response.json())