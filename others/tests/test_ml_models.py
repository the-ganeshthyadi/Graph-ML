import pytest
from src.ml_models import kmeans_predict, polynomial_regression_predict, knn_predict, knn_accuracy

def test_kmeans_predict():
    X = [[1], [2], [3], [4]]
    n_clusters = 2
    labels = kmeans_predict(X, n_clusters)
    assert len(labels) == len(X)

def test_polynomial_regression_predict():
    X = [[1], [2], [3]]
    y = [1, 4, 9]
    degree = 2
    predictions = polynomial_regression_predict(X, y, degree)
    assert len(predictions) == len(X)

def test_knn_predict():
    X_train = [[1], [2], [3]]
    y_train = [0, 1, 1]
    X_test = [[1.5]]
    n_neighbors = 2
    predictions = knn_predict(X_train, y_train, X_test, n_neighbors)
    assert len(predictions) == len(X_test)

def test_knn_accuracy():
    X_train = [[1], [2], [3]]
    y_train = [0, 1, 1]
    X_test = [[1.5]]
    y_test = [0]
    n_neighbors = 2
    accuracy = knn_accuracy(X_train, y_train, X_test, y_test, n_neighbors)
    assert isinstance(accuracy, float)
