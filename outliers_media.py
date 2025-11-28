import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("dados_limpos_media.csv")
df_out = df.copy()   


def plot_boxplot(data, column, title):
    plt.figure(figsize=(9,4))
    sns.boxplot(x=data[column])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"boxplot_media_{title.replace(' ','_')}.png")
    plt.close()


def iqr_clip(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    before_count = (
        (data[column] < lim_inf) |
        (data[column] > lim_sup)
    ).sum()

    data[column] = data[column].clip(lower=lim_inf, upper=lim_sup)

    print(f"[{column}] outliers tratados: {before_count} "
          f"(lim_inf={lim_inf:.2f}, lim_sup={lim_sup:.2f})")
    return data


print("\n=== Tratamento de Outliers ===\n")

for col in ["Km","Preco"]:
    # antes
    plot_boxplot(df_out, col, f"{col}_ANTES")
    # tratamento
    df_out = iqr_clip(df_out, col)
    # depois
    plot_boxplot(df_out, col, f"{col}_DEPOIS")


df_out.to_csv("dados_media_outliers_tratados.csv", index=False)
print(f"Total final de linhas: {df_out.shape[0]}")
