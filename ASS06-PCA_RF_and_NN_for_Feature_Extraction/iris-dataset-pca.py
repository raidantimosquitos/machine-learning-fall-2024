from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data  # Input samples
y = iris.target  # True labels (for visualization purposes)

# Step 1: Apply K-means clustering on the original data
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters for 3 iris classes
y_kmeans = kmeans.fit_predict(X)

# Step 2: Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize the clusters after PCA
plt.figure(figsize=(8, 6))
plt.grid(True)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.title('Iris Dataset Clusters with PCA-reduced Dimensions (After Clustering)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()