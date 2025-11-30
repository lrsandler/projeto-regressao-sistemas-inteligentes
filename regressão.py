import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
import joblib
import os


def treinar_modelos(nome_dataset, df, colunas_numericas, target):


    X = df[colunas_numericas].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pasta_saida = f"resultados_{nome_dataset}"
    os.makedirs(pasta_saida, exist_ok=True)

    joblib.dump(scaler, f"{pasta_saida}/scaler.pkl")


    resultados = {}

    # KNN Regressor ========================================================
    param_knn = {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }

    knn = GridSearchCV(KNeighborsRegressor(), param_knn, cv=5, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    y_pred_knn = knn.predict(X_test_scaled)

    resultados["KNN"] = {
        "best_params": knn.best_params_,
        "MAE": mean_absolute_error(y_test, y_pred_knn),
        "RMSE": mean_squared_error(y_test, y_pred_knn, squared=False),
        "R2": r2_score(y_test, y_pred_knn)
    }

    joblib.dump(knn.best_estimator_, f"{pasta_saida}/modelo_knn.pkl")


    # Random Forest ======================================================
    param_rf = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    rf = GridSearchCV(RandomForestRegressor(), param_rf, cv=5, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    y_pred_rf = rf.predict(X_test_scaled)

    resultados["RandomForest"] = {
        "best_params": rf.best_params_,
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": mean_squared_error(y_test, y_pred_rf, squared=False),
        "R2": r2_score(y_test, y_pred_rf)
    }

    joblib.dump(rf.best_estimator_, f"{pasta_saida}/modelo_random_forest.pkl")


    # Gradient Boosting ===================================================
    param_gb = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4]
    }

    gb = GridSearchCV(GradientBoostingRegressor(), param_gb, cv=5, n_jobs=-1)
    gb.fit(X_train_scaled, y_train)

    y_pred_gb = gb.predict(X_test_scaled)

    resultados["GradientBoosting"] = {
        "best_params": gb.best_params_,
        "MAE": mean_absolute_error(y_test, y_pred_gb),
        "RMSE": mean_squared_error(y_test, y_pred_gb, squared=False),
        "R2": r2_score(y_test, y_pred_gb)
    }

    joblib.dump(gb.best_estimator_, f"{pasta_saida}/modelo_gradient_boosting.pkl")



    df_resultados = pd.DataFrame(resultados).T
    df_resultados.to_csv(f"{pasta_saida}/resultados_grid_search.csv")

    print("\nResultados:")
    print(df_resultados)
    print(f"\nArquivos salvos na pasta: {pasta_saida}")




colunas_numericas = ["Ano", "Km", "Débitos", "Numero_proprietarios", "Airbags", "Couro", "Volume_motor"]

TARGET = "Preco"

df_knn = pd.read_csv("base_sem_anomalias_score_knn.csv")
df_media = pd.read_csv("base_sem_anomalias_score_media.csv")

treinar_modelos("knn", df_knn, colunas_numericas, TARGET)
treinar_modelos("media", df_media, colunas_numericas, TARGET)

