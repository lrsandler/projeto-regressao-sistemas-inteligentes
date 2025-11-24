import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, re
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import unicodedata

with open('config.json', 'r') as f:
    config = json.load(f)
data_path = config['dataset_path']
figures_path = config['save_figures_path']  
os.makedirs(figures_path, exist_ok=True)

df = pd.read_csv(data_path, encoding='utf-8', sep=',')

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

col_categoricas = ['Categoria', 'Combustivel', 'Tipo_cambio', 'Tração', 'Portas', 'Cor', 'Classificacao_Veiculo', 'Faixa_Preco']
for col in col_categoricas:
    df[col] = df[col].apply(padronizar_texto)


df = df.replace(['NA', ' ', ''], np.nan)
# caracteres não ASCII para nan
for col in df.columns:
    df[col] = df[col].apply(
        lambda x: np.nan if isinstance(x, str) and re.search(r'[^\x00-\x7F]', x) else x)

df['Débitos'] = df['Débitos'].replace('-', '0') #substitui - por 0
df['Débitos'] = pd.to_numeric(df['Débitos'], errors='coerce')

def limpar_km(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip()
    # remove tudo que não for dígito
    s = re.sub(r'[^\d]', '', s)
    if s == '':
        return np.nan
    return int(s)

df['Km'] = df['Km'].apply(limpar_km)

df['Volume_motor'] = df['Volume_motor'].astype(str).str.replace(r'[^\d.]', '', regex=True)
df['Volume_motor'] = pd.to_numeric(df['Volume_motor'], errors='coerce')
df.loc[df['Volume_motor'] < 0.5, 'Volume_motor'] = np.nan

# eliminar dados irrelevantes
colunas_ruido = ['ID' ,'Codigo_concessionaria', 'Rodas', 'Data_ultima_lavagem', 'Adesivos_personalizados', 'Radio_AM_FM', 'Historico_troca_oleo']
df_limpo = df.drop(columns=colunas_ruido)
features_df = df_limpo.drop(columns=['Preco'])
nulos_por_linha = features_df.isnull().sum(axis=1)
limite_nulos = 4
df_limpo = df_limpo[nulos_por_linha < limite_nulos].copy() 
print(f"Linhas removidas por excesso de nulos (>={limite_nulos}): {df.shape[0] - df_limpo.shape[0]}")

#retira dados com Preco igual a 0
condicao = df_limpo['Preco'].notna() & (df_limpo['Preco'] > 0)
df_sempreco = df_limpo.loc[~condicao]
df_limpo = df_limpo.loc[condicao]

df_limpo['Combustivel'] = df_limpo['Combustivel'].str.upper().str.strip()
df_limpo['Combustivel'] = df_limpo['Combustivel'].replace({
    'gasol.': 'gasolina',
    'dies.': 'diesel'})

# couro para binário
df_limpo['Couro'] = df_limpo['Couro'].map({'Sim': 1, 'Nao': 0})

df_limpo['Portas'] = df_limpo['Portas'].replace({'4-5': 4, '2-3': 2, '>5': 5})

df_limpo['Cor'] = df_limpo['Cor'].replace({'red': 'vermelho'})
df_limpo['Cor'] = df_limpo['Cor'].replace({'azul ceu': 'azul'})
df_limpo['Cor'] = df_limpo['Cor'].str.capitalize()

# tipos de colunas
colunas_numericas = ['Débitos', 'Couro', 'Ano', 'Volume_motor', 'Km', 'Cilindros', 'Airbags', 'Numero_proprietarios']
colunas_categoricas = ['Categoria', 'Combustivel', 'Tipo_cambio', 'Tração', 'Portas', 'Cor', 'Classificacao_Veiculo', 'Faixa_Preco']

def verificar_nan_por_col(df):
    nan_count = df.isna().sum()
    nan_percentage = (df.isna().sum() / len(df))
    return pd.DataFrame({'quantidade_nan': nan_count,
                        'percentual_nan': nan_percentage.round(2)}).sort_values(by='percentual_nan', ascending=False)

resultado_nan = verificar_nan_por_col(df)
print(f"Nulos no dataset original:\n {resultado_nan}\n")
resultado_nan_ = verificar_nan_por_col(df_limpo)
print(f"Nulos no dataset limpo:\n {resultado_nan_}\n")

def normalizar_minmax(df, colunas_numericas):
    "Normaliza as colunas numericas entre 0 e 1"
    scaler = MinMaxScaler()
    df_normalizado = df.copy()  
    df_normalizado[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])
    return df_normalizado, scaler

df_limpo_norm, scaler = normalizar_minmax(df_limpo, colunas_numericas)

def preencher_nulos_com_mediana_ou_moda(df, colunas_numericas, colunas_categoricas):
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

# usar norm ou nao??       
df_limpo_media = preencher_nulos_com_mediana_ou_moda(df_limpo, colunas_numericas, colunas_categoricas)

#lembrar de normalizar antes de preencher com knn
def preencher_nulos_com_knn(df, colunas_numericas, k=5):
    #explicar no relatorio pq faz sentido usar para cada atributo numerico

    imputer = KNNImputer(n_neighbors=k, missing_values=np.nan)
    df_numericas = df[colunas_numericas]
    df_numericas_imputadas = pd.DataFrame(imputer.fit_transform(df_numericas), columns=colunas_numericas, index=df.index)
    for col in colunas_numericas:
        df[col] = df_numericas_imputadas[col]
    return df

df_limpo_knn = preencher_nulos_com_knn(df_limpo_norm, colunas_numericas, k=5)

# manter colunas categóricas preenchidas com moda 
#alterar depois para arvore de decisao? testar variaveis categoricas e numericas com arvore 
df_limpo_knn[colunas_categoricas] = df_limpo_media[colunas_categoricas]

def verificar_unicos(df, name="valores_unicos.txt"):
    with open(name, 'w', encoding='utf-8') as f:
        f.write("Valores únicos por coluna:\n")
    for col in df.columns:
        valores_unicos = df[col].unique()
        #print(f"Coluna: {col}, Valores únicos: {valores_unicos}")
        with open(name, 'a', encoding='utf-8') as f:
            f.write(f"Coluna: {col}, Valores únicos: {valores_unicos}\n")

verificar_unicos(df, name="valores_unicos_original.txt")
verificar_unicos(df_limpo, name="valores_unicos.txt") 
verificar_unicos(df_limpo_media, name="valores_unicos_media.txt")
verificar_unicos(df_limpo_knn, name="valores_unicos_knn.txt")

print("tamanho do dataset original:", df.shape)
print("tamanho do dataset limpo:", df_limpo.shape)
print("tamanho do dataset sem preco:", df_sempreco.shape)

df_limpo_media.to_csv("dados_limpos_media.csv", index=False)
df_limpo_knn.to_csv("dados_limpos_knn.csv", index=False)
df_sempreco.to_csv("dados_sem_preco.csv", index=False)

#graficos -----------------------------------------------------------------
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

# matriz de correlação
correlacao = df_limpo_media.corr(numeric_only=True)
correlacao_knn = df_limpo_knn.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig(os.path.join(figures_path,"matriz_correlacao.png"))
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(correlacao_knn, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação com KNN')
plt.tight_layout()
plt.savefig(os.path.join(figures_path,"matriz_correlacao_knn.png"))
plt.show()
plt.close()     