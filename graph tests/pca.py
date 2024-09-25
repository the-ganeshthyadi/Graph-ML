import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load sample data (Iris dataset)
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Function to perform PCA and plot results
def plot_pca(X, y, feature_names, target_names):
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print(explained_variance_ratio)

    # Plot explained variance ratio
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    
    # Plot the explained variance ratio of each component
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
    plt.title('Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot the data projected onto the first two principal components
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title('PCA Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    
    # Beautify the plot
    sns.despine()
    plt.tight_layout()
    plt.show()

# Plot PCA results
plot_pca(X, y, feature_names, target_names)
