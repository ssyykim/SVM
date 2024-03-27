# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulated dataset
data = {
    'Age': [25, 34, 22, 27, 33, 29, 37, 36, 30, 31],
    'Annual Income (k$)': [40, 80, 30, 78, 82, 54, 88, 61, 55, 48],
    'Spending Score (1-100)': [61, 77, 40, 89, 91, 42, 95, 55, 50, 42]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Apply PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_pca)

# Plot the clustered data
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('Clusters of customers')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster')

# Show centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.legend()

plt.show()