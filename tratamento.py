import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
df = df.replace(['NA', ' ', ''], np.nan)

df['Débitos'] = df['Débitos'].replace('-', '0') #substitui - por 0
df['Débitos'] = pd.to_numeric(df['Débitos'], errors='coerce')

df['Km'] = df['Km'].astype(str).str.replace(' km', '', regex=False)
df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
df['Volume_motor'] = df['Volume_motor'].astype(str).str.replace(r'[^\d.]', '', regex=True)
df['Volume_motor'] = pd.to_numeric(df['Volume_motor'], errors='coerce')

# eliminar dados irrelevantes
colunas_ruido = ['ID', 'Codigo_concessionaria', 'Data_ultima_lavagem', 'Adesivos_personalizados', 'Radio_AM_FM', 'Historico_troca_oleo']
df_limpo = df.drop(columns=colunas_ruido)
features_df = df_limpo.drop(columns=['Preco'])
nulos_por_linha = features_df.isnull().sum(axis=1)
limite_nulos = 4
df_limpo = df_limpo[nulos_por_linha < limite_nulos].copy() 

# tipos de colunas
colunas_numericas = ['Débitos', 'Ano', 'Volume_motor', 'Km', 'Cilindros', 'Airbags', 'Numero_proprietarios']
colunas_categoricas = ['Categoria', 'Couro', 'Combustivel', 'Tipo_cambio', 'Tração', 'Portas', 'Rodas', 'Cor', 'Classificacao_Veiculo', 'Faixa_Preco']

# numéricas 
for col in colunas_numericas:
    if df_limpo[col].isnull().any():
        mediana = df_limpo[col].median()
        df_limpo[col].fillna(mediana, inplace=True)

# categóricas
for col in colunas_categoricas:
    if df_limpo[col].isnull().any():
        moda = df_limpo[col].mode()[0]
        df_limpo[col].fillna(moda, inplace=True)
        print(f"Categorica: '{col}' preenchida com a Moda ('{moda}').")

df_limpo['Fabricante'].fillna(df_limpo['Fabricante'].mode()[0], inplace=True)
df_limpo['Modelo'].fillna(df_limpo['Modelo'].mode()[0], inplace=True)

df_limpo.to_csv("dados_limpos.csv", index=False)