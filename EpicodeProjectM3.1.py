 #PARTE 1 - DATA EXPLORATION E PREPROCESSING

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

#Caricamento Dataset e Analisi
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(df.describe())


#Standardizzazione
from sklearn.preprocessing import StandardScaler
df_scaled = StandardScaler().fit_transform(df)

#PARTE 2 - STIMA DI K CON DBSCAN

from sklearn.cluster import DBSCAN

#Parametri iterazione
eps_values = [0.35, 0.45, 0.55]
min_samples = 5

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df_scaled)

#Calcolo numero clusters
n_clusters = len(set(labels) - {-1})

#Calcolo punti rumorosi 
n_noise = list(labels).count(-1)

print("eps:", eps)
print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)
print()

#Analyzing the DBSCAN Output
#At eps = 0.35, the density threshold is very strict. We identify 4 clusters but a high number of points (104) are classified as noise.
#At eps = 0.45, DBSCAN flags 54 points as noise and separates the remaining density regions into 3 clusters.
#At eps = 0.55, the density reach is larger, merging regions into 2 clusters with only 29 noise points.
#Because the raw Iris dataset naturally consists of 3 distinct biological species (Setosa, Versicolor, and Virginica), the DBSCAN run with eps = 0.45 suggests that  K=3  is a highly plausible number of clusters for this dataset based on density partitioning.


#PARTE 3 - STIMA DI K CON ELBOW METHOD

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Definisco K range
inertias = []
k_range = range(1,11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_) 

#Elbow method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o', linestyle='-', color='b', linewidth=2)

plt.axvline(x=3, color='r', linestyle='--', label='Elbow Point (K = 3)')

plt.title('The Elbow Method for Optimal K', fontsize=14)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (WCSS)', fontsize=12)
plt.xticks(k_range)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()

#PARTE 4 - APPLICAZIONE FINALE E VISUALIZZAZIONE

from sklearn.metrics import silhouette_score

# Kmean
optimal_k = 2

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(df_scaled)

# Calcolo punteggio Silhouette 
score = silhouette_score(df_scaled, cluster_labels)
print(f"Silhouette Score for K = {optimal_k}: {score:.4f}")

# Visulizzazione dati 2D
plt.figure(figsize=(10, 6))

plt.scatter(
    df_scaled[:, 0],
    df_scaled[:, 1],
    c=cluster_labels,
    cmap='viridis'
    )

# Cetroidi
centroids = final_kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color='red',
    marker='X',
    s=250,
    label='Centroids'
)

plt.xlabel(df.columns[0] + ' (Standardized)')
plt.ylabel(df.columns[1] + ' (Standardized)')
plt.title(f'K-Means Clustering (K = {optimal_k})')
plt.legend()
plt.show()