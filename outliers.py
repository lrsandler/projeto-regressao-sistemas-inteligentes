import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_boxplot(data, column, title, folder_path):
    os.makedirs(folder_path, exist_ok=True) 
    plt.figure(figsize=(9,4))
    sns.boxplot(x=data[column])
    plt.title(title)
    plt.tight_layout()
    file_path = os.path.join(folder_path, f"boxplot_{title.replace(' ', '_')}.png")
    plt.savefig(file_path)
    plt.close()

def iqr_clip(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    before_count = ((data[column] < lim_inf) | (data[column] > lim_sup)).sum()

    data[column] = data[column].clip(lower=lim_inf, upper=lim_sup)

    print(f"[{column}] outliers tratados: {before_count} (lim_inf={lim_inf:.2f}, lim_sup={lim_sup:.2f})")
    return data

def tratar_outliers(df, colunas, boxplot_path):
    for col in colunas:
        if col in df.columns:
            plot_boxplot(df, col, f"{col}_ANTES", boxplot_path)
            df = iqr_clip(df, col)
            plot_boxplot(df, col, f"{col}_DEPOIS", boxplot_path)
    return df
