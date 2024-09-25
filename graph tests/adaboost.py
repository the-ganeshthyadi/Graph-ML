import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

def adaboost_classifier(X, y, n_estimators=50, learning_rate=1.0):
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    model.fit(X, y)
    return model

def plot_adaboost_feature_importances(X, y, params, ax):
    n_estimators = params.get('n_estimators', 50)
    learning_rate = params.get('learning_rate', 1.0)
    model = adaboost_classifier(X, y, n_estimators, learning_rate)

    # Visualization: Plot feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = params.get('feature_names', [f'Feature {i}' for i in range(X.shape[1])])
    
    # Ensure that the number of feature names matches the number of features
    if len(feature_names) < X.shape[1]:
        feature_names.extend([f'Feature {i}' for i in range(len(feature_names), X.shape[1])])
    
    # Plot
    ax.bar(range(X.shape[1]), importances[indices], align='center', color='skyblue', edgecolor='black')
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('AdaBoost Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Sample Data
X, y = make_classification(n_samples=100, n_features=6, n_informative=4, n_redundant=1, n_clusters_per_class=1, random_state=42)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
params = {
    'n_estimators': 50,
    'learning_rate': 1.0,
    'feature_names': ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E', 'Feature F']
}
plot_adaboost_feature_importances(X, y, params, ax)
plt.tight_layout()
plt.show()
