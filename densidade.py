import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors



colunas_numericas = ['Ano', 'Km', 'Couro', 'Numero_proprietarios', 'Airbags',
                     'Volume_motor', 'Preco', 'Débitos']

df = pd.read_csv("dados_limpos_knn.csv", encoding='utf-8', sep=',')
X = df[colunas_numericas].values

# Redução de dimensionalidade para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#isolation forest
iforest = IsolationForest(contamination=0.05, random_state=42)
y_if = iforest.fit_predict(X)
score_if = -iforest.score_samples(X)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sc1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=score_if, cmap="turbo", s=15, edgecolor='k')
plt.title("Isolation Forest - PCA")
plt.colorbar(sc1, label="Score")

plt.subplot(1, 2, 2)
plt.plot(np.sort(score_if)[::-1], 'k.', markersize=3)
plt.title("Curva de Score - Isolation Forest")
plt.xlabel("Amostras ordenadas")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

#lof
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_lof = lof.fit_predict(X)
score_lof = -lof.negative_outlier_factor_

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sc2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=score_lof, cmap="viridis", s=15, edgecolor='k')
plt.title("LOF - PCA")
plt.colorbar(sc2, label="Score")

plt.subplot(1, 2, 2)
plt.plot(np.sort(score_lof)[::-1], 'k.', markersize=3)
plt.title("Curva de Score - LOF")
plt.xlabel("Amostras ordenadas")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

""" 
#grafico p determinar k para lof
n = X.shape[0]
minPts = np.arange(7, 125)  # valores de k que desejamos analisar

score_lof_k = np.zeros((len(minPts), n))

for i, k in enumerate(minPts):
    lof_k = LocalOutlierFactor( n_neighbors=k )
    lof_k.fit(X)
    score_lof_k[i, :] = -lof_k.negative_outlier_factor_


for j in range(n):
    plt.plot(minPts, score_lof_k[:, j], linewidth=0.7, c='gray', alpha=0.5)
# plt.axvline(x = 22, color = 'orange', ls='--')
# plt.axvline(x = 35, color = 'orange', ls='--')
# plt.axhline(y=3.25, color='r', linestyle='--', label='Threshold')
# plt.axhline(y=1.9, color='gold', linestyle='--', label='Threshold')
# plt.axhline(y=1.4, color='limegreen', linestyle='--', label='Threshold')
plt.gca().set_xlabel('k'); plt.gca().set_ylabel('LOF')
plt.tight_layout()
plt.savefig('thresh.pdf')
plt.show() """


# Aplicar DBSCAN com eps escolhido e min_samples
dbscan = DBSCAN(eps=1.5, min_samples=20)
y_db = dbscan.fit_predict(X)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_db, cmap="coolwarm", s=15, edgecolor='k')
plt.title("DBSCAN - PCA")

plt.subplot(1, 2, 2)
counts = pd.Series(y_db).value_counts()
plt.bar(counts.index.astype(str), counts.values)
plt.title("Distribuição dos clusters / outliers")
plt.xlabel("Cluster (-1 = Ruído)")
plt.ylabel("Quantidade")

plt.tight_layout()
plt.show()

#contagem de outliers
print("\nNúmero de outliers detectados:")
print(f"Isolation Forest: {np.sum(y_if == -1)}")
print(f"LOF: {np.sum(y_lof == -1)}")
print(f"DBSCAN (ruído): {np.sum(y_db == -1)}")
