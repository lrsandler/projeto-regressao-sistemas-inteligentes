# Projeto de Regressão - Sistemas Inteligentes

## Visão Geral

Este projeto implementa um pipeline completo de análise de regressão para previsão de preços de veículos. O sistema inclui tratamento de dados, detecção de anomalias, treinamento de múltiplos modelos de machine learning, e uma funcionalidade completa de dashboard para visualização e monitoramento de resultados.

## Características Principais

- **Tratamento Avançado de Dados**: Limpeza, padronização e imputação de valores faltantes
- **Detecção de Outliers**: Métodos IQR e Isolation Forest
- **Múltiplos Modelos**: Linear Regression, KNN, Random Forest, Gradient Boosting
- **Dashboard de Visualização**: Sistema completo de visualizações para análise de dados e desempenho de modelos
- **Duas Estratégias de Imputação**: Comparação entre KNN e Média/Moda

## Estrutura do Projeto

```
projeto-regressao-sistemas-inteligentes/
│
├── tratamento.py           # Módulo de tratamento e limpeza de dados
├── outliers.py            # Funções para detecção e tratamento de outliers
├── isolation_forest.py    # Detecção de anomalias usando Isolation Forest
├── regressão.py           # Treinamento e avaliação de modelos
├── config.json            # Configurações do projeto
│
├── figures/               # Visualizações geradas
│   └── boxplots/         # Boxplots de variáveis numéricas
│
├── resultados_knn/        # Resultados com imputação KNN
└── resultados_media/      # Resultados com imputação Média/Moda
```

## Instalação

### Requisitos

- Python 3.8+
- Bibliotecas necessárias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Configuração

Edite o arquivo `config.json` para configurar os caminhos:

```json
{
    "dataset_path": "../data/train.csv",
    "save_figures_path": "./figures/",
    "boxplot_path": "./figures/boxplots/"
}
```

## Como Usar

### 1. Tratamento de Dados

Execute o módulo de tratamento para limpar e preparar os dados:

```bash
python tratamento.py
```

**Saídas:**
- `dados_limpos_knn.csv` - Dados limpos com imputação KNN
- `dados_limpos_media.csv` - Dados limpos com imputação por média/moda
- Boxplots em `./figures/boxplots/`

**Nota:** Para gerar visualizações adicionais (distribuição de preços, matrizes de correlação, pair plots), descomente as linhas 283-286 no arquivo `tratamento.py` antes de executar.

### 2. Detecção de Anomalias

Execute a detecção de anomalias com Isolation Forest:

```bash
python isolation_forest.py
```

**Saídas:**
- `base_sem_anomalias_score_knn.csv` - Base limpa sem anomalias (KNN)
- `base_sem_anomalias_score_media.csv` - Base limpa sem anomalias (Média)
- Visualizações 3D de outliers

### 3. Treinamento de Modelos

Execute o treinamento e avaliação dos modelos:

```bash
python regressão.py
```

**Saídas:**
- Modelos treinados (`.pkl`)
- Métricas de desempenho (`.csv`)
- Gráficos comparativos em `./resultados_*/`

## Dashboard de Visualização

Este projeto inclui um **sistema completo de dashboard** para visualização e análise de resultados. 

### Visualizações Disponíveis

#### 📊 Análise de Dados
- Distribuição de preços
- Matrizes de correlação (Spearman e Pearson)
- Pair plots para análise multivariada
- Boxplots para detecção de outliers

#### 🎯 Detecção de Anomalias
- Gráficos 3D de outliers
- Curvas de score para ajuste de threshold
- Estatísticas de anomalias detectadas

#### 🏆 Desempenho de Modelos
- Comparação de RMSE entre modelos
- Comparação de R² Score
- Gráficos Predito vs. Real
- Tabela de métricas detalhadas

### Documentação Completa do Dashboard

Para uma explicação detalhada de todas as funcionalidades do dashboard, consulte:

**[📖 DASHBOARD_DOCUMENTATION.md](./DASHBOARD_DOCUMENTATION.md)**

Este documento inclui:
- Descrição completa de cada visualização
- Casos de uso e interpretação
- Fluxo de dados através do sistema
- Guia de customização
- Exemplos de código

## Modelos Implementados

### 1. Linear Regression
Modelo base para comparação

### 2. KNN Regressor
- Busca em grid: n_neighbors, weights, p
- Cross-validation: 5 folds

### 3. Random Forest
- Busca em grid: n_estimators, max_depth, min_samples_split
- Cross-validation: 5 folds

### 4. Gradient Boosting
- Busca em grid: n_estimators, learning_rate, max_depth
- Cross-validation: 5 folds

## Métricas de Avaliação

Todos os modelos são avaliados usando:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** Score

## Variáveis do Dataset

### Variáveis Numéricas
- `Ano` - Ano do veículo
- `Km` - Quilometragem
- `Débitos` - Débitos associados
- `Numero_proprietarios` - Número de proprietários anteriores
- `Airbags` - Quantidade de airbags
- `Volume_motor` - Volume do motor
- `Cilindros` - Número de cilindros
- `Preco` - Preço (variável alvo)

### Variáveis Categóricas
- `Categoria` - Categoria do veículo
- `Fabricante` - Fabricante
- `Modelo` - Modelo
- `Couro` - Interior de couro
- `Combustivel` - Tipo de combustível
- `Tipo_cambio` - Tipo de câmbio
- `Tração` - Tipo de tração
- `Portas` - Número de portas
- `Cor` - Cor do veículo
- `Classificacao_Veiculo` - Classificação
- `Faixa_Preco` - Faixa de preço

## Pipeline de Processamento

```
Dados Brutos
    ↓
Limpeza e Padronização
    ↓
Tratamento de Valores Faltantes (KNN / Média-Moda)
    ↓
Detecção de Outliers (IQR)
    ↓
Detecção de Anomalias (Isolation Forest)
    ↓
Dados Limpos
    ↓
Normalização e Encoding
    ↓
Treinamento de Modelos
    ↓
Avaliação e Comparação
    ↓
Modelo Final
```

## Resultados

Os resultados dos modelos são salvos em:
- `resultados_knn/` - Resultados usando imputação KNN
- `resultados_media/` - Resultados usando imputação por média/moda

Cada pasta contém:
- Gráficos de comparação (RMSE, R²)
- Gráficos predito vs. real para cada modelo
- CSV com métricas detalhadas
- Modelos treinados (.pkl)
- Scaler para normalização (.pkl)

## Contribuindo

Para adicionar novas funcionalidades ao dashboard ou melhorar o pipeline:

1. Consulte `DASHBOARD_DOCUMENTATION.md` para entender a estrutura atual
2. Siga os padrões de código existentes
3. Adicione visualizações em um módulo apropriado
4. Atualize a documentação

## Licença

Este projeto é um trabalho acadêmico para a disciplina de Sistemas Inteligentes.

## Autores

Desenvolvido como projeto de regressão para análise de preços de veículos.

## Suporte

Para dúvidas sobre:
- **Funcionalidades do Dashboard**: Consulte `DASHBOARD_DOCUMENTATION.md`
- **Uso dos módulos**: Veja os comentários no código-fonte
- **Configuração**: Verifique `config.json`
