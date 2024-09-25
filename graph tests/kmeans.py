import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Sample data
X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])

# Apply KMeans
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Scatter plot of data points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=100, edgecolor='k', legend='full', marker='o')

# Scatter plot of centroids
sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color='red', s=200, marker='X', label='Centroids')

# Customizing plot
plt.title('K-Means Clustering', fontsize=16)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.legend(title='Cluster', title_fontsize='13', fontsize='11')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()
