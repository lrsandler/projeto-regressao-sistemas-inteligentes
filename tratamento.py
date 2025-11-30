import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, re
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import unicodedata
from sklearn.neighbors import NearestNeighbors
from outliers import plot_boxplot, tratar_outliers

with open('config.json', 'r') as f:
    config = json.load(f)
data_path = config['dataset_path']
figures_path = config['save_figures_path']  
boxplot_path = config['boxplot_path']
os.makedirs(boxplot_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

def padronizar_texto(s):
    if isinstance(s, str):
        s = s.strip().lower()
        # normaliza caracteres acentuados
        s = unicodedata.normalize('NFD', s)
        # remove acentos
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        s = s.replace('ç', 'c')
        # remove múltiplos espaços
        s = re.sub(r'\s+', ' ', s)
        return s
    return s

def limpar_km(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip()
    # remove tudo que não for dígito
    s = re.sub(r'[^\d]', '', s)
    if s == '':
        return np.nan
    return int(s)

def verificar_nan_por_col(df):
    nan_count = df.isna().sum()
    nan_percentage = (df.isna().sum() / len(df))
    return pd.DataFrame({'quantidade_nan': nan_count,
                        'percentual_nan': nan_percentage.round(2)}).sort_values(by='percentual_nan', ascending=False)

def normalizar(df, colunas_numericas):
    scaler = StandardScaler()
    df_normalizado = df.copy()  
    df_normalizado[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])
    return df_normalizado, scaler

def preencher_nan_com_media_moda(df, colunas_numericas, colunas_categoricas):
    # numéricas 
    for col in colunas_numericas:
        if df[col].isnull().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            print(f"Numerica: '{col}' preenchida com a mediana '{mediana}'")
    # categóricas        
    for col in colunas_categoricas:
        if df[col].isnull().any():
            moda = df[col].mode()[0]
            df[col]= df[col].fillna(moda)
            print(f"Categorica: '{col}' preenchida com a Moda '{moda}'")
    return df 

#lembrar de normalizar antes de preencher com knn
def verificar_densidade_knn(df, colunas_numericas, k=5):
    df_num = df[colunas_numericas].dropna()

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(df_num)
    dist, _ = nn.kneighbors(df_num)

    plt.hist(dist.mean(axis=1), bins=30)
    plt.title("Distribuição das distâncias médias entre vizinhos")
    plt.xlabel("Distância média")
    plt.ylabel("Frequência")
    plt.show()

    print(f"Média das distâncias: {dist.mean():.4f}")

def preencher_nulos_com_knn(df, colunas_numericas, k=5, mostrar_graficos=True):
    #explicar no relatorio pq faz sentido usar para cada atributo numerico

    df_numericas = df[colunas_numericas]

    #Estatísticas antes da imputação
    stats_antes = df_numericas.describe().T

    imputer = KNNImputer(n_neighbors=k, missing_values=np.nan)
    df_numericas_imputadas = pd.DataFrame(
        imputer.fit_transform(df_numericas),
        columns=colunas_numericas,
        index=df.index
    )

    mascara_imputacao = df_numericas.isna()  # True para onde valores foram preenchidos
    linhas_imputadas = mascara_imputacao.any(axis=1).sum()
    colunas_imputadas = mascara_imputacao.any()

    # Substituindo os valores
    df[colunas_numericas] = df_numericas_imputadas

    stats_depois = df[colunas_numericas].describe().T
    print(f"\nNúmero de linhas imputadas: {linhas_imputadas}")

    print("\n Variação estatística:")
    relatorio_stats = pd.DataFrame({
        "Media antes": stats_antes["mean"],
        "Media depois": stats_depois["mean"],
        "Std antes": stats_antes["std"],
        "Std depois": stats_depois["std"]
    })
    print(relatorio_stats)

    #verifica distribuição antes e depois
    if mostrar_graficos:
        for col in colunas_imputadas[colunas_imputadas].index:
            plt.figure(figsize=(10, 4))
            
            # Antes da imputação
            plt.subplot(1, 2, 1)
            plt.hist(df_numericas[col].dropna(), bins=30, alpha=0.7)
            plt.title(f"{col} - Antes da Imputação")
            plt.xlabel(col)
            plt.ylabel("Frequência")

            # Depois da imputação
            plt.subplot(1, 2, 2)
            plt.hist(df_numericas_imputadas[col].dropna(), bins=30, alpha=0.7)
            plt.title(f"{col} - Depois da Imputação (KNN)")
            plt.xlabel(col)
            plt.ylabel("Frequência")

            plt.suptitle(f"Distribuição antes vs depois - {col}", fontsize=12)

            plt.tight_layout()
            plt.show()
            plt.close() 

    return df

def verificar_unicos(df, name="valores_unicos.txt"):
    with open(name, 'w', encoding='utf-8') as f:
        f.write("Valores únicos por coluna:\n")
    for col in df.columns:
        valores_unicos = df[col].unique()
        #print(f"Coluna: {col}, Valores únicos: {valores_unicos}")
        with open(name, 'a', encoding='utf-8') as f:
            f.write(f"Coluna: {col}, Valores únicos: {valores_unicos}\n")

#graficos -----------------------------------------------------------------
def plotar_distribuicao_preco(df_limpo):
    plt.figure(figsize=(10, 6))
    plt.hist(df_limpo['Preco'], bins=30, color='blue', edgecolor='black')
    plt.title('Distribuição de Preço dos Veículos')
    plt.xlim(0, df_limpo['Preco'].quantile(0.999)) 
    plt.xlabel('Preço')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, "distribuicao_preco.png"))
    plt.show()
    plt.close()

def matriz_correlacao(df_limpo_knn):
    correlacao_knn = df_limpo_knn.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlacao_knn, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path,"matriz_correlacao.png"))
    plt.show()
    plt.close()

def main():

    df = pd.read_csv(data_path, encoding='utf-8', sep=',')

    # eliminar dados irrelevantes
    colunas_ruido = ['ID' ,'Codigo_concessionaria', 'Rodas', 'Data_ultima_lavagem', 'Adesivos_personalizados', 'Radio_AM_FM', 'Historico_troca_oleo']
    df = df.drop(columns=colunas_ruido)

    # tipos de colunas
    colunas_numericas = ['Débitos', 'Ano', 'Volume_motor', 'Km', 'Cilindros', 'Airbags', 'Numero_proprietarios']
    colunas_categoricas = ['Categoria', 'Couro', 'Combustivel', 'Tipo_cambio', 'Tração', 'Portas', 'Cor', 'Classificacao_Veiculo', 'Faixa_Preco']

    for col in colunas_categoricas:
        df[col] = df[col].apply(padronizar_texto)

    df = df.replace(['NA', ' ', ''], np.nan)
    # caracteres não ASCII para nan
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: np.nan if isinstance(x, str) and re.search(r'[^\x00-\x7F]', x) else x)

    df['Débitos'] = df['Débitos'].replace('-', '0') #substitui - por 0
    df['Débitos'] = pd.to_numeric(df['Débitos'], errors='coerce')

    df['Km'] = df['Km'].apply(limpar_km)

    df['Volume_motor'] = df['Volume_motor'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Volume_motor'] = pd.to_numeric(df['Volume_motor'], errors='coerce')
    df.loc[df['Volume_motor'] < 0.5, 'Volume_motor'] = np.nan


    features_df = df.drop(columns=['Preco'])
    nulos_por_linha = features_df.isnull().sum(axis=1)
    limite_nulos = 3
    df_limpo = df[nulos_por_linha < limite_nulos].copy() 
    print(f"Linhas removidas por excesso de nulos (>={limite_nulos}): {df.shape[0] - df_limpo.shape[0]}")

    #retira dados com Preco igual a 0
    condicao = df_limpo['Preco'].notna() & (df_limpo['Preco'] > 0)
    #df_sempreco = df_limpo.loc[~condicao]
    df_limpo = df_limpo.loc[condicao]

    #padroniza combustivel e cor
    df_limpo['Combustivel'] = df_limpo['Combustivel'].str.lower().str.strip()
    df_limpo['Combustivel'] = df_limpo['Combustivel'].replace({
        'gasol.': 'gasolina',
        'dies.': 'diesel'})

    df_limpo['Cor'] = df_limpo['Cor'].replace({'red': 'vermelho'})
    df_limpo['Cor'] = df_limpo['Cor'].replace({'azul ceu': 'azul'})
    df_limpo['Cor'] = df_limpo['Cor'].str.capitalize()

    # couro para binário
    #df_limpo['Couro'] = df_limpo['Couro'].map({'Sim': 1, 'Nao': 0})

    #verificar nan 
    resultado_nan = verificar_nan_por_col(df)
    print(f"Nulos no dataset original:\n {resultado_nan}\n")
    resultado_nan_ = verificar_nan_por_col(df_limpo)
    print(f"Nulos no dataset limpo:\n {resultado_nan_}\n")

    #plota boxplots iniciais
    for col in colunas_numericas:
        plot_boxplot(df_limpo, col, f"{col}_ANTES_TRATAMENTO", boxplot_path)

    df_limpo_media = preencher_nan_com_media_moda(df_limpo, colunas_numericas, colunas_categoricas)

    #normalizar antes de knn
    df_limpo_norm, scaler = normalizar(df_limpo_media, colunas_numericas)
    #verfica a densidade antes de preencher com knn
    verificar_densidade_knn(df_limpo_norm, colunas_numericas, k=5)

    df_limpo_knn = preencher_nulos_com_knn(df_limpo_norm, colunas_numericas, k=5, mostrar_graficos=True)
    
    #inverter a normalização depois de inputar
    df_limpo_knn[colunas_numericas] = scaler.inverse_transform(df_limpo_knn[colunas_numericas])

    # manter colunas categóricas preenchidas com moda  
    df_limpo_knn[colunas_categoricas] = df_limpo_media[colunas_categoricas]

    #tratar outliers com iqr
    colunas_tratamento_iqr = ['Km', 'Preco']
    df_limpo_knn_iqr = tratar_outliers(df_limpo_knn, colunas_tratamento_iqr, boxplot_path)
    df_limpo_media_iqr = tratar_outliers(df_limpo_media, colunas_tratamento_iqr, boxplot_path)


    print("tamanho do dataset original:", df.shape)
    print("tamanho do dataset limpo:", df_limpo.shape)
    print("tamanho do dataset limpo media:", df_limpo_media_iqr.shape)
    print("tamanho do dataset limpo knn:", df_limpo_knn_iqr.shape)

    df_limpo_media.to_csv("dados_limpos_media.csv", index=False)
    df_limpo_knn_iqr.to_csv("dados_limpos_knn.csv", index=False)

    verificar_unicos(df, name="valores_unicos_original.txt")
    verificar_unicos(df_limpo, name="valores_unicos.txt") 
    verificar_unicos(df_limpo_media_iqr, name="valores_unicos_media.txt")
    verificar_unicos(df_limpo_knn_iqr, name="valores_unicos_knn.txt")

    #plotar_distribuicao_preco(df_limpo)
    #matriz_correlacao(df_limpo_knn)

if __name__ == "__main__":
    main()