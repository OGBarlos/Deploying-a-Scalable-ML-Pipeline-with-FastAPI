import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
url_get = "http://127.0.0.1:8000/"
r_get = requests.get(url_get)

# TODO: print the status code
print("GET Status Code:", r_get.status_code)
# TODO: print the welcome message
print("GET Welcome Message:", r_get.json()["message"])

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
url_post = "http://127.0.0.1:8000/data/"
r_post = requests.post(url_post, json=data)

# TODO: print the status code
print("POST Status Code:", r_post.status_code)
# TODO: print the result
print("POST Result:", r_post.json()["result"])
