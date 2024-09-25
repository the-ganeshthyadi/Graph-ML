import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

def gradient_boosting_classifier(X, y, n_estimators=100, learning_rate=0.1):
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    model.fit(X, y)
    return model

def plot_feature_importances(X, y, params):
    # Train the Gradient Boosting model
    model = gradient_boosting_classifier(X, y, params.get('n_estimators', 100), params.get('learning_rate', 0.1))
    
    # Extract feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), importances[indices], align='center', color='skyblue')
    plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in indices], rotation=45, ha='right')
    plt.title('Gradient Boosting Feature Importances', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    
    # Beautify the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Sample data
X = np.array([[1.2, 2.3],
              [3.4, 4.5],
              [5.6, 6.7],
              [7.8, 8.9],
              [9.0, 1.1]])

y = np.array([0, 1, 0, 1, 0])

# Parameters for gradient boosting
params = {
    'n_estimators': 100,
    'learning_rate': 0.1
}

# Generate and beautify the feature importance plot
plot_feature_importances(X, y, params)
