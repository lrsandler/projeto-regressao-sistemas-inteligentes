import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def treinar_modelos(nome_dataset, df, colunas_numericas, colunas_categoricas, target):
    # Separar numéricas e categóricas
    X_num = df[colunas_numericas].values
    X_cat = pd.get_dummies(df[colunas_categoricas], drop_first=True).values
    y = df[target].values

    # Normalizar apenas as numéricas
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Concatenar com as categóricas one-hot
    X = np.hstack([X_num_scaled, X_cat])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criar pasta para resultados
    pasta_saida = f"resultados_{nome_dataset}"
    os.makedirs(pasta_saida, exist_ok=True)

    joblib.dump(scaler, f"{pasta_saida}/scaler.pkl")

    resultados = {}

    # ------------------ Linear Regression ------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    resultados["LinearRegression"] = {
        "best_params": "N/A",
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "RMSE": root_mean_squared_error(y_test, y_pred_lr),
        "R2": r2_score(y_test, y_pred_lr)
    }
    joblib.dump(lr, f"{pasta_saida}/modelo_linear_regression.pkl")

    # ------------------ KNN Regressor ----------------------
    param_knn = {
        "n_neighbors": [3,5,7,9,11,15],
        "weights": ["uniform","distance"],
        "p": [1,2]
    }

    grid_knn = GridSearchCV(KNeighborsRegressor(), param_knn, cv=5, n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    best_knn = grid_knn.best_estimator_
    y_pred_knn = best_knn.predict(X_test)

    resultados["KNN"] = {
        "best_params": grid_knn.best_params_,
        "MAE": mean_absolute_error(y_test, y_pred_knn),
        "RMSE": root_mean_squared_error(y_test, y_pred_knn),
        "R2": r2_score(y_test, y_pred_knn)
    }
    joblib.dump(best_knn, f"{pasta_saida}/modelo_knn.pkl")

    # ------------------ Random Forest ---------------------
    param_rf = {
        "n_estimators": [100,200],
        "max_depth": [None,10,20],
        "min_samples_split": [2,5]
    }

    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_rf, cv=5, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    resultados["RandomForest"] = {
        "best_params": grid_rf.best_params_,
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": root_mean_squared_error(y_test, y_pred_rf),
        "R2": r2_score(y_test, y_pred_rf)
    }
    joblib.dump(best_rf, f"{pasta_saida}/modelo_random_forest.pkl")

    # ------------------ Gradient Boosting -----------------
    param_gb = {
        "n_estimators": [100,200],
        "learning_rate": [0.01,0.05,0.1],
        "max_depth": [2,3,4]
    }

    grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_gb, cv=5, n_jobs=-1)
    grid_gb.fit(X_train, y_train)
    best_gb = grid_gb.best_estimator_
    y_pred_gb = best_gb.predict(X_test)

    resultados["GradientBoosting"] = {
        "best_params": grid_gb.best_params_,
        "MAE": mean_absolute_error(y_test, y_pred_gb),
        "RMSE": root_mean_squared_error(y_test, y_pred_gb),
        "R2": r2_score(y_test, y_pred_gb)
    }
    joblib.dump(best_gb, f"{pasta_saida}/modelo_gradient_boosting.pkl")

    # ----------------- Salvar e mostrar resultados -----------------
    df_resultados = pd.DataFrame(resultados).T
    df_resultados.to_csv(f"{pasta_saida}/resultados_grid_search.csv")
    print(df_resultados)

    # ----------------- Gráficos -----------------
    # RMSE
    plt.figure(figsize=(8,5))
    plt.bar(df_resultados.index, df_resultados['RMSE'], color='skyblue')
    plt.ylabel('RMSE')
    plt.title('Comparação de RMSE entre modelos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/comparacao_rmse.png")
    plt.close()

    # R2
    plt.figure(figsize=(8,5))
    plt.bar(df_resultados.index, df_resultados['R2'], color='salmon')
    plt.ylabel('R2 Score')
    plt.title('Comparação de R2 entre modelos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/comparacao_r2.png")
    plt.close()

    # Predito vs Real
    y_preds = {
        'LinearRegression': y_pred_lr,
        'KNN': y_pred_knn,
        'RandomForest': y_pred_rf,
        'GradientBoosting': y_pred_gb
    }

    for name, y_pred in y_preds.items():
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Preço Real")
        plt.ylabel("Preço Predito")
        plt.title(f"{name} - Predito vs Real")
        plt.tight_layout()
        plt.savefig(f"{pasta_saida}/predito_vs_real_{name}.png")
        plt.close()


colunas_numericas = ["Ano", "Km", "Débitos", "Numero_proprietarios", "Airbags", "Volume_motor"]
colunas_categoricas = [
    'Categoria', 'Fabricante', 'Modelo', 'Couro','Combustivel',
    'Tipo_cambio','Tração','Portas','Cor','Classificacao_Veiculo','Faixa_Preco'
]
TARGET = "Preco"

df_knn = pd.read_csv("base_sem_anomalias_score_knn.csv")
df_media = pd.read_csv("base_sem_anomalias_score_media.csv")

# Retirar colunas não desejadas
df_knn = df_knn.drop(columns=["is_outlier", "score_if", "Cilindros"], errors='ignore')
df_media = df_media.drop(columns=["is_outlier", "score_if", "Cilindros"], errors='ignore')

#treinar_modelos("knn", df_knn, colunas_numericas, colunas_categoricas, TARGET)
#treinar_modelos("media", df_media, colunas_numericas, colunas_categoricas, TARGET)

diff = df_knn["Km"] - df_media["Km"]
print(diff[diff != 0].head())
print("Qtde diferente:", (diff != 0).sum())
