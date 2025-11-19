#FOMLEXP9.a
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Step 1: Load dataset
dataset = pd.read_csv('/Mall_Customers.csv')  # ← Change to your correct path
X = dataset.iloc[:, [3, 4]].values

print(dataset.head())

# Step 2: Elbow method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=0
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# ✅ Graph 1 — Elbow Method
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Step 3: Apply k-means with 5 clusters
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=0
)
y_kmeans = kmeans.fit_predict(X)

# ✅ Graph 2 — All clusters together
plt.figure(figsize=(6, 4))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids')
plt.title('Clusters of Customers (All)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()

# ✅ Graph 3 — Individual clusters
for i in range(5):
    plt.figure(figsize=(6, 4))
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, c='yellow', label='Centroids')
    plt.title(f'Cluster {i+1}')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1–100)')
    plt.legend()
    plt.show()

# ✅ Graph 4 — Cluster comparison (pairplot style)
plt.figure(figsize=(6, 4))
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=50, alpha=0.6, label=f'Cluster {i+1}')
plt.title('Comparison of All Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()
