import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib

df_knn = pd.read_csv("dados_limpos_knn.csv", encoding='utf-8')
df_media = pd.read_csv("dados_limpos_media.csv", encoding='utf-8')

colunas_numericas = ['Ano','Km','Preco' ]

X_num_knn = df_knn[colunas_numericas]
X_num_media = df_media[colunas_numericas]
#print(X_num_knn.head())

# Normalizar para o Isolation Forest
scaler_media = StandardScaler()
scaler_media.fit(X_num_media)  
scaler_knn = StandardScaler()
scaler_knn.fit(X_num_knn)
 
X_norm_media = scaler_media.transform(X_num_media) 
X_norm_knn = scaler_knn.fit_transform(X_num_knn)

contamination = 200/ len(df_knn)  # proporção de outliers 
isoforest = IsolationForest(contamination=contamination, n_estimators=100, random_state=42)

y_if_media = isoforest.fit_predict(X_norm_media)       # -1 é outlier
score_if_media = -isoforest.score_samples(X_norm_media)
df_media["is_outlier"] = (y_if_media == -1)
df_media["score_if"] = score_if_media
print(f"Número de outliers detectados: {np.sum(df_media['is_outlier'])}")

y_if_knn = isoforest.fit_predict(X_norm_knn)           # -1 é outlier
score_if_knn = -isoforest.score_samples(X_norm_knn)  
df_knn["is_outlier"] = (y_if_knn == -1)
df_knn["score_if"] = score_if_knn
print(f"Número de outliers detectados: {np.sum(df_knn['is_outlier'])}")

df_norm_media = pd.DataFrame(X_norm_media, columns=colunas_numericas)
df_norm_knn = pd.DataFrame(X_norm_knn, columns=colunas_numericas)

def plot_outliers(df, df_norm, score_if):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    seq = [df[i] for i in colunas_numericas] 

    scatter = ax.scatter(seq[0], seq[1], seq[2],
                        c=df['is_outlier'], cmap="coolwarm", s=20)

    ax.set_title("Outliers detectados pelo Isolation Forest")
    ax.set_xlabel(seq[0].name)
    ax.set_ylabel(seq[1].name)
    ax.set_zlabel(seq[2].name)
    plt.show()

    #curva score para ajudar a escolher contamination
    plt.figure(figsize=(8,4))
    score_sorted = np.sort(score_if)[::-1]
    # posições brutais no eixo X p destacar
    posicoes = [50, 100, 200, 300]
    plt.plot(score_sorted, 'k', label='Score ordenado')
    for p in posicoes:
        if p < len(score_sorted): 
            plt.axvline(p, linestyle='--', linewidth=0.8)
            plt.scatter(p, score_sorted[p], color='red', s=50)  # marcador
            plt.annotate(f"x={p}", (p, score_sorted[p]),
                        textcoords="offset points", xytext=(0, 8), ha='center')

    plt.title("Curva de Score - Isolation Forest")
    plt.ylabel("Score")
    plt.xlabel("Amostras ordenadas")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_outliers(df_media, df_norm_media, score_if_media)
plot_outliers(df_knn, df_norm_knn, score_if_knn)

# Mostrar head dos mais anomolos 
top_outliers = df_knn.sort_values(by="score_if", ascending=False).head(10).copy()
print("Top outliers knn:\n", top_outliers[['Ano', 'Km', 'Preco', 'score_if']])
top_outliers_media = df_media.sort_values(by="score_if", ascending=False).head(10).copy()
print("Top outliers media:\n", top_outliers_media[['Ano', 'Km', 'Preco', 'score_if']])

#lim score de anomaia
limite_score = 0.65
anomalias_knn = df_knn[df_knn["score_if"] > limite_score].copy()
anomalias_media = df_media[df_media["score_if"] > limite_score].copy()

df_knn_sem_anomalias = df_knn[df_knn["score_if"] <= limite_score].copy()
df_media_sem_anomalias = df_media[df_media["score_if"] <= limite_score].copy()

anomalias_knn.to_csv(f"anomalias_score_acima_knn.csv", index=False, encoding='utf-8')
anomalias_media.to_csv(f"anomalias_score_acima_media.csv", index=False, encoding='utf-8')

df_knn_sem_anomalias.to_csv(f"base_sem_anomalias_score_knn.csv", index=False, encoding='utf-8')
df_media_sem_anomalias.to_csv(f"base_sem_anomalias_score_media.csv", index=False, encoding='utf-8')

print(f"Total de anomalias removidas (KNN): {len(anomalias_knn)}")
print(f"Total de anomalias removidas (Média): {len(anomalias_media)}")
