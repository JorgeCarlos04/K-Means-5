import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Carga y preparación de datos
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Escalado para mejorar rendimiento de K-means[1]

# 2. Método del codo
def analisis_codo(datos, max_clusters=10):
    wcss = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(datos)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, max_clusters+1), wcss, 'bx-')
    plt.xlabel('Número de Clústeres (k)')
    plt.ylabel('WCSS')
    plt.title('Método del Codo')

# 3. Análisis de silueta
def analisis_silueta(datos, max_clusters=10):
    silhouette_scores = []
    for k in range(2, max_clusters+1):  # Silueta no aplica para k=1[2]
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(datos)
        score = silhouette_score(datos, labels)
        silhouette_scores.append(score)
    
    plt.subplot(1,2,2)
    plt.plot(range(2, max_clusters+1), silhouette_scores, 'ro-')
    plt.xlabel('Número de Clústeres (k)')
    plt.ylabel('Puntaje Silueta')
    plt.title('Análisis de Silueta')
    plt.tight_layout()
    plt.show()

# Ejecución completa
analisis_codo(X_scaled)
analisis_silueta(X_scaled)
