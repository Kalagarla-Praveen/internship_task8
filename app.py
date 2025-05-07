import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Optional: drop non-numeric or irrelevant columns
df = df.drop(['CustomerID', 'Gender'], axis=1)  # Adjust based on actual columns

# Step 2: Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Optional: Apply PCA for 2D visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Step 3: Elbow Method to find optimal K
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method For Optimal K')
plt.show()

# Step 4: Fit KMeans with optimal K (e.g., K=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Step 5: Visualize clusters
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-Means Clustering (PCA-reduced)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Step 6: Evaluate using Silhouette Score
sil_score = silhouette_score(scaled_data, cluster_labels)
print(f"Silhouette Score: {sil_score:.3f}")
