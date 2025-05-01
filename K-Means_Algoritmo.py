import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Cargar y preprocesar datos
data = load_iris()
X = StandardScaler().fit_transform(data.data)  

# Variables k a evaluar 
valores = range(2, 11)
inertias = []
silhouette_scores = []

# Calcular los valores  para cada k
for k in valores:
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# Grafico funcion (plt.)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, marker='o', linestyle='--')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inertia')
plt.title('Método del Codo')
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Coeficiente de Silueta')
plt.title('Análisis de Silueta')
plt.tight_layout()
plt.show()

# K optimo o proximo
optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
print(f'k óptimo (silueta): {optimal_k_silhouette}')