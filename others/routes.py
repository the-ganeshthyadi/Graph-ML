from flask import Blueprint, request, jsonify
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from ml_models import (
    kmeans_predict, polynomial_regression_predict, knn_predict, knn_accuracy,
    linear_regression_predict, decision_tree_classifier, random_forest_classifier,
    svm_classifier, naive_bayes_classifier, gradient_boosting_classifier,
    adaboost_classifier, neural_network_classifier, xgboost_classifier, pca_transform
)

ml_predict = Blueprint('ml_predict', __name__)

@ml_predict.route('/ml-predict', methods=['POST'])
def ml_predict_route():
    data = request.json
    model_type = data.get('model_type')
    X = np.array(data.get('X'))
    params = data.get('params', {})
    y = np.array(data.get('y')) if 'y' in data else None
    graph_params = data.get('graph_params', {})
    
    graph_type = graph_params.get('type', 'line')
    graph_color = graph_params.get('color', 'blue')
    graph_linewidth = graph_params.get('linewidth', 1)
    graph_background_color = graph_params.get('background_color', 'white')
    return_values = graph_params.get('return_values', False)
    colors = graph_params.get('colors', [graph_color])

    fig, ax = plt.subplots()
    fig.patch.set_facecolor(graph_background_color)

    calculated_values = None

    if model_type == 'kmeans':
        n_clusters = params.get('n_clusters', 2)
        labels, centers = kmeans_predict(X, n_clusters)
        calculated_values = labels.tolist()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k')
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centroids')
        ax.legend()
        ax.set_title('KMeans Clustering')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    elif model_type == 'polynomial_regression':
        degree = params.get('degree', 2)
        predictions, coef, intercept = polynomial_regression_predict(X, y, degree)
        calculated_values = predictions.tolist()
        ax.plot(X, predictions, color=colors[0], linewidth=graph_linewidth)
        ax.set_title(f'Polynomial Regression (Degree {degree})')
        ax.set_xlabel('X')
        ax.set_ylabel('Predicted Y')
    elif model_type == 'knn':
        X_train = np.array(data.get('X_train'))
        y_train = np.array(data.get('y_train'))
        X_test = np.array(data.get('X_test'))
        n_neighbors = params.get('n_neighbors', 5)
        predictions, train_accuracy = knn_predict(X_train, y_train, X_test, n_neighbors)
        calculated_values = predictions.tolist()
        ax.scatter(X_test[:, 0], predictions, color=colors[0])
        ax.set_title(f'KNN Predictions (Accuracy {train_accuracy:.2f})')
        ax.set_xlabel('X Test')
        ax.set_ylabel('Predictions')
    elif model_type == 'knn_accuracy':
        X_train = np.array(data.get('X_train'))
        y_train = np.array(data.get('y_train'))
        X_test = np.array(data.get('X_test'))
        y_test = np.array(data.get('y_test'))
        n_neighbors = params.get('n_neighbors', 5)
        accuracy = knn_accuracy(X_train, y_train, X_test, y_test, n_neighbors)
        calculated_values = accuracy
        ax.text(0.5, 0.5, f'Accuracy: {accuracy:.2f}', fontsize=12, ha='center')
        ax.set_title('KNN Accuracy')
        ax.axis('off')
    elif model_type == 'linear_regression':
        predictions, coef, intercept = linear_regression_predict(X, y)
        calculated_values = predictions.tolist()
        ax.plot(X, predictions, color=colors[0], linewidth=graph_linewidth)
        ax.set_title('Linear Regression')
        ax.set_xlabel('X')
        ax.set_ylabel('Predicted Y')
    elif model_type == 'decision_tree':
        max_depth = params.get('max_depth', None)
        model = decision_tree_classifier(X, y, max_depth)
        # Visualization: Plot the tree
        from sklearn.tree import plot_tree
        plot_tree(model, filled=True)
        ax.set_title('Decision Tree')
    elif model_type == 'random_forest':
        n_estimators = params.get('n_estimators', 100)
        model = random_forest_classifier(X, y, n_estimators)
        # Visualization: Plot feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ax.bar(range(X.shape[1]), importances[indices], align='center')
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels([f'Feature {i}' for i in indices])
        ax.set_title('Random Forest Feature Importances')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
    elif model_type == 'svm':
        C = params.get('C', 1.0)
        kernel = params.get('kernel', 'rbf')
        model = svm_classifier(X, y, C, kernel)
        # Visualization: Plot decision boundaries
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap='coolwarm')
        ax.set_title('SVM Decision Boundary')
    elif model_type == 'naive_bayes':
        model = naive_bayes_classifier(X, y)
        # Visualization: Plot class distribution
        classes = np.unique(y)
        counts = np.array([np.sum(y == cls) for cls in classes])
        ax.bar(classes, counts, color=colors[0])
        ax.set_title('Naive Bayes Class Distribution')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Count')
    elif model_type == 'gradient_boosting':
        n_estimators = params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        model = gradient_boosting_classifier(X, y, n_estimators, learning_rate)
        # Visualization: Plot feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ax.bar(range(X.shape[1]), importances[indices], align='center')
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels([f'Feature {i}' for i in indices])
        ax.set_title('Gradient Boosting Feature Importances')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
    elif model_type == 'adaboost':
        n_estimators = params.get('n_estimators', 50)
        learning_rate = params.get('learning_rate', 1.0)
        model = adaboost_classifier(X, y, n_estimators, learning_rate)
        # Visualization: Plot feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ax.bar(range(X.shape[1]), importances[indices], align='center')
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels([f'Feature {i}' for i in indices])
        ax.set_title('AdaBoost Feature Importances')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
    elif model_type == 'neural_network':
        hidden_layer_sizes = params.get('hidden_layer_sizes', (100,))
        activation = params.get('activation', 'relu')
        model = neural_network_classifier(X, y, hidden_layer_sizes, activation)
        # Visualization: Plot training progress
        # Note: MLPClassifier does not return training history, so this example is conceptual
        ax.text(0.5, 0.5, 'Neural Network Model', fontsize=12, ha='center')
        ax.axis('off')
    elif model_type == 'xgboost':
        n_estimators = params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        model = xgboost_classifier(X, y, n_estimators, learning_rate)
        # Visualization: Plot feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ax.bar(range(X.shape[1]), importances[indices], align='center')
        ax.set_xticks(range(X.shape[1]))
        ax.set_xticklabels([f'Feature {i}' for i in indices])
        ax.set_title('XGBoost Feature Importances')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
    elif model_type == 'pca':
        n_components = params.get('n_components', 2)
        transformed_X = pca_transform(X, n_components)
        if n_components == 2:
            ax.scatter(transformed_X[:, 0], transformed_X[:, 1])
            ax.set_title('PCA Result')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
        else:
            ax.text(0.5, 0.5, 'PCA with more than 2 components', fontsize=12, ha='center')
            ax.axis('off')
    else:
        return jsonify({'error': 'Invalid model type'}), 400

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return jsonify({
        'image': img_str,
        'calculated_values': calculated_values if return_values else None
    })
