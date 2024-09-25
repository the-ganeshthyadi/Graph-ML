## API Usage
1. Using KMeans
    ```"model_type": "kmeans",
    "X": [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]],
    "params": {
        "n_clusters": 2  # Example parameter for K-Means
    },
    "graph_params": {
        "type": "scatter",
        "color": "blue",
        "background_color": "white",
        "return_values": True,
        "xlabel": "sample1",  # X-axis label
        "ylabel": "sample2",  # Y-axis label
        "title": "K-Means"  # Plot title
    }```


2. Using Polynomial Regression

    ```"model_type": "polynomial_regression",
    "X": [[1.0], [1.5], [5.0], [8.0], [1.0], [9.0]],  # Sample feature data
    "y": [0, 0, 1, 1, 0, 1],  # Sample target data
    "params": {
        "degree": 3  # Polynomial degree for the regression
    },
    "graph_params": {
        "type": "line",  # For polynomial regression, a line plot is used
        "color": "blue",  # Color of the polynomial regression line
        "background_color": "white",  # Background color of the plot
        "xlabel": "X-axis Label",  # X-axis label
        "ylabel": "Y-axis Label",  # Y-axis label
        "title": "Polynomial Regression Plot"  # Title of the plot
    },
    "return_values":"True"```


3. Using KNN

    ```    "model_type": "knn",
    'X_train': [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],  # Training features
    'y_train': [0, 0, 1, 1, 1],  # Training labels
    'X_test': [[1.5, 2.5], [3.5, 4.5]],
    "params": {
        "n_neighbors": 4  # Polynomial degree for the regression
    },
    "graph_params": {
    'xlabel': 'Feature 1',
    'ylabel': 'Predicted Class',
    'title': 'KNN Classification'
    },
    "return_values":"True"
    ```


4. Using Decision Tree

    ```"model_type": "decision_tree",
        "X": [
        [2.3, 1.5],
        [1.2, 3.4],
        [3.7, 2.1],
        [2.0, 2.9],
        [0.8, 0.5],
        [3.2, 1.8],
        [2.8, 2.2],
        [1.9, 3.1],
        [3.5, 0.9],
        [0.6, 2.7]
        ],
        "y": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    "params": {
        "max_depth": 3
    },
    "graph_params": {
        "xlabel": "Feature 1",
        "ylabel": "Feature 2",
        "title": "Decision Tree Classifier Example",
        "class_names": ["Class 0", "Class 1"]
    }
    ```

5. Using Random Forest

    ```  "model_type": "random_forest",
        "X": X.tolist(),  # Convert numpy array to list for JSON serialization
        "y": y.tolist(),  # Convert numpy array to list for JSON serialization
        "params": {
            "n_estimators": 100,  # Number of trees in the forest
            "max_depth": 3  # Maximum depth of the trees
        },
        "graph_params": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "feature_importance":{
                "label1":"x",
                "label2":"y"
            },
            "decision_boundry":{
                "feature1":"feature 1",
                "feature2":"feature 2"
            },
            "feature_importance_title": "Random Forest Feature Importances Example",
            "decision_boundry_title": "Random Forest Decision Boundaryr Example"
        }
        ```

6. Using Gradient Boosting

    ```    "model_type": "gradient_boosting",
        "X": X.tolist(),  # Convert numpy array to list for JSON serialization
        "y": y.tolist(),
    "params": {
        'n_estimators': 100,
        'learning_rate': 0.1
    },
    "graph_params": {
        "title": "Naive Bayes Classifier Example",
        "xlabel": "Feature 1",
        "ylabel": "Feature 2",
        "feature_names": ['Feature A', 'Feature B']
    }
    ```


7. Using Adaboost

    ```  "model_type": "adaboost",
        'X': X_list,
        'y': y_list,
    "params": {
        'n_estimators': 50,
        'learning_rate': 1.0,
    },
    "graph_params": {
        "title": "AdaBoost Feature Importances",
        "xlabel": "Features",
        "ylabel": "Importance",
        'feature_names': ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E', 'Feature F']
    }
    ```


8. Using Neural Network

    ```    'model_type': 'neural_network',
        "params":{
            'hidden_layer_sizes': (10, 5),
            'activation': 'tanh'
        },  # Using the 'tanh' activation function
        'X': [[0.1, 0.2], [0.4, 0.6], [0.8, 0.5], [0.3, 0.7], [0.9, 0.2]],  # Example feature data (5 samples, 2 features)
        'y': [0, 1, 0, 1, 0],
        "graph_params":{
            "xlabel": "Label 1",
            "ylabel": "Label 2",
            "title": "Title"
        }```


9. Using XGBoost

   ``` "model_type": "xgboost",
    "X": X.tolist(),
    "y": y.tolist(), 
    "params": {
        "n_estimators": 150,
        "learning_rate": 0.05
    },
    "graph_params": {
        "title": "AdaBoost Feature Importances",
        "xlabel": "Features",
        "ylabel": "Importance",
        'feature_names': ['Feature A', 'Feature B', 'Feature C']
    }```


10. Using PCA

    ```    "model_type": "pca",
        "X": X.tolist(),
        "y": y.tolist(),
        "params": {
            "n_components": 2
        },
        "graph_params":{
            "title" : "Title",
            "xlabel": "Label1",
            "ylabel": "Label2"
        }```