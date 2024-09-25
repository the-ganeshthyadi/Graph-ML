import requests
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.datasets import load_iris, make_classification


# API endpoint
url = "http://127.0.0.1:5000/ml-predict"



data1 = load_iris()
X = data1.data
y = data1.target

data = {
    "model_type": "logistic_regression",
    "X": [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3],
    [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1]
    ],
    "y": [0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    "params": {
        "C": 1.0,
        "max_iter": 100,
        "solver": "lbfgs"
    }
}

# Sending the POST request
response = requests.post(url, json=data)

# Handling the response
response_data = response.json()

if 'calculated_values' in response_data:
    print("Calculated Values:")
    print(response_data['calculated_values'])
else:
    print("No return values found in the response.")


# Check if the graph is in the response
if 'graph' in response_data:
    # Decode the base64-encoded image
    image_data = base64.b64decode(response_data['graph'])
    image = Image.open(BytesIO(image_data))
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axes
    plt.show()
else:
    print("No graph returned in the response.")

