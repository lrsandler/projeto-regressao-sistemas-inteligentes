import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("dados_limpos.csv")

# preço faltante
df_predict = df[df['Preco'].isnull()].copy()
# preço conhecido
df_train = df[df['Preco'].notnull()].copy()

# Boxplot
def plot_boxplot(data, column, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"boxplot_{column}.png")
    plt.close()

# tratamento de outliers usando o método IQR 
def iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lim_superior = Q3 + 1.5 * IQR
    
    # valores acima do limite superior são substituídos pelo limite
    outliers_count = data[data[column] > lim_superior].shape[0]
    data[column] = np.where(data[column] > lim_superior, lim_superior, data[column])
    
    print(f"Coluna '{column}': {outliers_count} outliers acima do limite superior limitados a {lim_superior:.2f}")
    return data

# aplica no conjunto Preco e Km do conjunto de treino
df_train = iqr(df_train, 'Preco')
df_train = iqr(df_train, 'Km')

# Boxplots após o tratamento
plot_boxplot(df_train, 'Preco', 'Boxplot de Preco (Após o Capping)')
plot_boxplot(df_train, 'Km', 'Boxplot de Km (Após o Capping)')

# salvar os dados
df_final_limpo = pd.concat([df_train, df_predict], ignore_index=True)
df_final_limpo.to_csv("dados_com_outliers_tratados.csv", index=False)