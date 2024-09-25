import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# Generate a synthetic dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM model
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)

# Create a mesh grid for plotting the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict the decision boundary
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the decision boundary
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

# Plot the training and test points
scatter_train = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=100, label='Training Data')
scatter_test = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=100, marker='x', label='Test Data')

# Set plot title and labels
ax.set_title('SVM Decision Boundary with Training and Test Data', fontsize=16)
ax.set_xlabel('Feature 1', fontsize=14)
ax.set_ylabel('Feature 2', fontsize=14)

# Add a legend
ax.legend(handles=[scatter_train, scatter_test], loc='upper right', fontsize=12)

# Beautify the axes
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.5)

# Display the plot
plt.show()
