import sys
import os
import pytest
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from __init__ import create_app

@pytest.fixture
def client():
    app = create_app()
    app.testing = True
    with app.test_client() as client:
        yield client

def test_ml_predict(client):
    response = client.post('/ml-predict', json={
        'model_type': 'kmeans',
        'X': [[1], [2], [3], [4]],
        'params': {'n_clusters': 2}
    })
    data = json.loads(response.data)
    assert 'labels' in data

def test_generate_graph(client):
    response = client.post('/generate-graph', json={
        'graph_params': {
            'type': 'line',
            'color': 'red',
            'linewidth': 2,
            'background_color': 'lightgrey',
            'x': [1, 2, 3],
            'y': [1, 4, 9]
        }
    })
    data = json.loads(response.data)
    assert 'image' in data
