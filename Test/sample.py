import requests
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use the first two features for simplicity (sepal length and width)
y = iris.target

# API endpoint
url = "http://127.0.0.1:5000/ml-predict"

# Data to be sent to the API
data = {
    "model_type": "kmeans",
    "X": X.tolist(),
    "params": {
        "n_clusters": 3  # Number of clusters for K-Means, matching the 3 classes in the Iris dataset
    },
    "graph_params": {
        "type": "scatter",
        "color": "blue",
        "background_color": "white",
        "return_values": True,
        "xlabel": "Sepal Length",  # X-axis label
        "ylabel": "Sepal Width",  # Y-axis label
        "title": "K-Means on Iris Dataset"  # Plot title
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
