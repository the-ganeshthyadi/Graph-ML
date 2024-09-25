import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap

def neural_network_classifier(X, y, hidden_layer_sizes=(100,), activation='relu', solver='adam'):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, random_state=0)
    model.fit(X, y)
    return model

def plot_decision_boundary(X, y, model, ax, title='Neural Network Decision Boundary'):
    # Set up the mesh grid
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and scatter plot
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_points = ListedColormap(['#FF0000', '#0000FF'])
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_background)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap_points)
    
    # Adding plot details
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.legend(*scatter.legend_elements(), title='Classes')
    ax.grid(True, linestyle='--', alpha=0.7)

# Sample Data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Train Neural Network Model
model = neural_network_classifier(X, y, hidden_layer_sizes=(10, 10), activation='relu')

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_decision_boundary(X, y, model, ax)
plt.tight_layout()
plt.show()
