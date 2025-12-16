# Dashboard and Visualization Functionality

## Overview

This project implements a comprehensive visualization and analysis dashboard for vehicle price regression analysis. The dashboard functionality is distributed across multiple Python modules, each providing specialized visualization and reporting capabilities for different stages of the data analysis pipeline.

## Dashboard Components

### 1. Data Treatment Dashboard (`tratamento.py`)

The data treatment module provides several visualization functions to monitor data quality and preprocessing steps:

#### 1.1 Price Distribution Visualization
**Function:** `plotar_distribuicao_preco(df_limpo)`

Creates a histogram showing the distribution of vehicle prices in the dataset.

**Output:**
- **File:** `./figures/distribuicao_preco.png`
- **Description:** Displays the frequency distribution of vehicle prices, trimmed at the 99.9th percentile for better visualization
- **Use Case:** Understand the price range and distribution patterns in your dataset

#### 1.2 Correlation Matrix Dashboard
**Function:** `matriz_correlacao(df_limpo_knn)`

Generates two correlation heatmaps to analyze relationships between numerical variables.

**Outputs:**
- **Spearman Correlation:** `./figures/matriz_correlacao_spearman.png`
  - Best for non-linear relationships and discrete variables (Doors, Airbags, Number of owners)
  - Shows correlation between all numerical features
  
- **Pearson Correlation:** `./figures/matriz_correlacao_pearson.png`
  - Best for linear relationships
  - Excludes discrete variables (Doors, Airbags, Number of owners, Cylinders) for accurate analysis

**Use Case:** Identify which features are most correlated with vehicle price and detect multicollinearity

#### 1.3 Pair Plot Visualization
**Function:** `plot_pairplot(df, colunas_numericas, hue_col=None, log_transform=None, save_path=None)`

Creates a comprehensive pair plot showing relationships between multiple numerical features.

**Parameters:**
- `df`: DataFrame with the data
- `colunas_numericas`: List of numerical columns to plot
- `hue_col`: Optional categorical column for color coding
- `log_transform`: List of columns to apply log transformation for better visualization
- `save_path`: Path to save the figure

**Output:**
- **File:** Customizable (e.g., `figures/pairplot.png`)
- **Description:** Matrix of scatter plots showing pairwise relationships, with KDE plots on the diagonal

**Use Case:** Explore multivariate relationships and patterns across different price ranges

#### 1.4 Boxplot Dashboard
**Function:** `plot_boxplot(data, column, title, folder_path)` (from `outliers.py`)

Creates boxplots for numerical variables to visualize outliers and data distribution.

**Outputs:**
- **Before Treatment:** `./figures/boxplots/boxplot_{column}_ANTES_TRATAMENTO.png`
- **After Treatment:** `./figures/boxplots/boxplot_{column}_DEPOIS.png`

**Columns Monitored:**
- Year (Ano)
- Mileage (Km)
- Debts (Débitos)
- Number of owners (Numero_proprietarios)
- Airbags
- Engine volume (Volume_motor)
- Price (Preco)

**Use Case:** Identify outliers and verify the effectiveness of outlier treatment methods

#### 1.5 KNN Density Verification
**Function:** `verificar_densidade_knn(df, colunas_numericas, k=5)`

Visualizes the distribution of distances between nearest neighbors to assess data density.

**Output:**
- **Display:** Interactive histogram showing distance distribution
- **Console:** Average distance statistics

**Use Case:** Validate that data is dense enough for KNN imputation to work effectively

### 2. Outlier Detection Dashboard (`isolation_forest.py`)

The Isolation Forest module provides advanced visualization for anomaly detection:

#### 2.1 3D Outlier Visualization
**Function:** `plot_outliers(df, df_norm, score_if)`

Creates an interactive 3D scatter plot showing detected outliers.

**Features:**
- **Axes:** Year, Mileage, Price
- **Color coding:** Normal points vs. outliers (detected by Isolation Forest)
- **Interactive:** Can be rotated and zoomed

**Output:** Interactive matplotlib 3D plot

**Use Case:** Visually inspect which data points are identified as outliers in 3D space

#### 2.2 Anomaly Score Dashboard
**Function:** `plot_outliers(df, df_norm, score_if)` (score plot component)

Displays the distribution of anomaly scores to help choose the contamination threshold.

**Features:**
- Sorted anomaly scores in descending order
- Vertical lines at positions 50, 100, 200, and 300
- Red markers highlighting potential cutoff points
- Helps determine optimal number of outliers to remove

**Output:** Line plot with annotations

**Use Case:** Determine the optimal threshold for identifying true anomalies

**Console Outputs:**
- Number of outliers detected
- Top 10 most anomalous records with their scores

### 3. Model Performance Dashboard (`regressão.py`)

The regression module creates a comprehensive dashboard for model comparison:

#### 3.1 RMSE Comparison Chart
**Function:** `treinar_modelos()` (generates automatically)

Bar chart comparing Root Mean Squared Error across all models.

**Output:**
- **File:** `./resultados_{dataset}/comparacao_rmse.png` (where `{dataset}` is either 'knn' or 'media' based on the imputation strategy used)
- **Models Compared:**
  - Linear Regression
  - KNN Regressor
  - Random Forest
  - Gradient Boosting

**Use Case:** Identify which model has the lowest prediction error

#### 3.2 R² Score Comparison Chart
**Function:** `treinar_modelos()` (generates automatically)

Bar chart comparing R² scores to show model fit quality.

**Output:**
- **File:** `./resultados_{dataset}/comparacao_r2.png` (where `{dataset}` is either 'knn' or 'media' based on the imputation strategy used)

**Use Case:** Determine which model explains the most variance in the target variable

#### 3.3 Predicted vs. Actual Price Plots
**Function:** `treinar_modelos()` (generates for each model)

Scatter plots comparing predicted prices against actual prices.

**Outputs:** One plot per model (where `{dataset}` is either 'knn' or 'media')
- `./resultados_{dataset}/predito_vs_real_LinearRegression.png`
- `./resultados_{dataset}/predito_vs_real_KNN.png`
- `./resultados_{dataset}/predito_vs_real_RandomForest.png`
- `./resultados_{dataset}/predito_vs_real_GradientBoosting.png`

**Features:**
- Scatter points: Each prediction
- Red dashed line: Perfect prediction line (y=x)
- Closer points to the line = better predictions

**Use Case:** Visually assess prediction accuracy and identify systematic biases

#### 3.4 Model Performance Report
**Output:**
- **File:** `./resultados_{dataset}/resultados_grid_search.csv` (where `{dataset}` is either 'knn' or 'media')
- **Console:** Formatted table display

**Content:**
- Best hyperparameters for each model
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

**Use Case:** Comprehensive numerical comparison of all models

## Data Flow Through the Dashboard

```
Raw Data (train.csv)
    ↓
[Data Treatment Module]
    ├── Cleaning and standardization
    ├── Missing value imputation (KNN/Mean-Mode)
    ├── Outlier detection (IQR method)
    └── Dashboard Output: Boxplots, distributions, correlations
    ↓
Cleaned Data (dados_limpos_knn.csv / dados_limpos_media.csv)
    ↓
[Isolation Forest Module]
    ├── Anomaly detection using Isolation Forest
    ├── Score calculation and threshold determination
    └── Dashboard Output: 3D outlier plots, score curves
    ↓
Data without Anomalies (base_sem_anomalias_score_knn.csv)
    ↓
[Regression Module]
    ├── Model training (Linear, KNN, RF, GB)
    ├── Hyperparameter tuning with GridSearchCV
    ├── Performance evaluation
    └── Dashboard Output: Comparison charts, prediction plots, performance metrics
    ↓
Final Model & Results (modelo_*.pkl, resultados_*.csv)
```

## Configuration

The dashboard uses a centralized configuration file:

**File:** `config.json`

```json
{
    "dataset_path": "../data/train.csv",
    "save_figures_path": "./figures/",
    "boxplot_path": "./figures/boxplots/"
}
```

**Parameters:**
- `dataset_path`: Location of the input CSV file
- `save_figures_path`: Root directory for all visualization outputs
- `boxplot_path`: Specific directory for boxplot visualizations

## How to Use the Dashboard

### Step 1: Data Treatment and Initial Visualization

```python
# Run the treatment module to generate cleaning dashboards
python tratamento.py
```

**Generated Visualizations:**
- Boxplots for all numerical features (before/after outlier treatment)
- KNN density histogram
- Unique values report (valores_unicos*.txt)

### Step 2: Anomaly Detection Dashboard

```python
# Run isolation forest to generate anomaly detection visualizations
python isolation_forest.py
```

**Generated Visualizations:**
- 3D scatter plots showing outliers
- Anomaly score curves with threshold markers
- Console output with top anomalies

### Step 3: Model Performance Dashboard

```python
# Uncomment the training lines in regressão.py
# treinar_modelos("knn", df_knn, colunas_numericas, colunas_categoricas, TARGET)
# treinar_modelos("media", df_media, colunas_numericas, colunas_categoricas, TARGET)

python regressão.py
```

**Generated Visualizations:**
- RMSE comparison bar chart
- R² comparison bar chart
- Predicted vs. actual scatter plots (4 models)
- Performance metrics CSV

**Note:** The first parameter to `treinar_modelos()` ('knn' or 'media') determines the output directory name (`resultados_knn/` or `resultados_media/`) where all visualizations and models will be saved. This allows you to compare results from both imputation strategies side by side.

## Key Features

### 1. **Comprehensive Data Quality Monitoring**
- Visualize data distributions at every stage
- Track the impact of cleaning and preprocessing
- Identify data quality issues early

### 2. **Advanced Outlier Detection**
- Multiple methods: IQR and Isolation Forest
- Visual confirmation of detected outliers
- Flexible threshold adjustment

### 3. **Model Comparison Dashboard**
- Side-by-side comparison of multiple models
- Visual and numerical performance metrics
- Easy identification of best-performing model

### 4. **Reproducibility**
- All visualizations are saved to disk
- Consistent file naming convention
- Configuration-driven paths

### 5. **Two Imputation Strategies**
- Compare KNN vs. Mean/Mode imputation
- Separate visualization pipelines
- Independent model training and evaluation

## Output Directory Structure

```
projeto-regressao-sistemas-inteligentes/
│
├── figures/
│   ├── distribuicao_preco.png
│   ├── matriz_correlacao_spearman.png
│   ├── matriz_correlacao_pearson.png
│   ├── pairplot.png
│   └── boxplots/
│       ├── boxplot_Ano_ANTES_TRATAMENTO.png
│       ├── boxplot_Ano_DEPOIS.png
│       ├── boxplot_Km_ANTES_TRATAMENTO.png
│       ├── boxplot_Km_DEPOIS.png
│       └── ... (more boxplots)
│
├── resultados_knn/
│   ├── comparacao_rmse.png
│   ├── comparacao_r2.png
│   ├── predito_vs_real_LinearRegression.png
│   ├── predito_vs_real_KNN.png
│   ├── predito_vs_real_RandomForest.png
│   ├── predito_vs_real_GradientBoosting.png
│   ├── resultados_grid_search.csv
│   ├── scaler.pkl
│   └── modelo_*.pkl (trained models)
│
└── resultados_media/
    └── (same structure as resultados_knn/)
```

## Technical Details

### Visualization Libraries
- **matplotlib**: Core plotting library for all charts
- **seaborn**: Statistical visualizations (heatmaps, boxplots, pair plots)
- **mpl_toolkits.mplot3d**: 3D scatter plots for outlier detection

### Data Processing
- **pandas**: Data manipulation and CSV I/O
- **numpy**: Numerical operations
- **sklearn**: Preprocessing, imputation, and model training

### Best Practices Implemented
1. **Consistent styling**: All plots use appropriate figure sizes and tight layouts
2. **File management**: Automatic directory creation with `os.makedirs()`
3. **Resource cleanup**: `plt.close()` after saving to prevent memory leaks
4. **Reproducibility**: Fixed random seeds (random_state=42)
5. **Validation**: Statistical reports before and after imputation

## Customization Guide

### Adding New Visualizations

To add a new visualization to the dashboard:

1. Create a function following the naming pattern:
```python
def plot_custom_analysis(df, save_path):
    plt.figure(figsize=(10, 6))
    # Your plotting code here
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

2. Call it in the appropriate module:
```python
plot_custom_analysis(df, os.path.join(figures_path, "custom_plot.png"))
```

### Modifying Output Paths

Edit `config.json` to change where visualizations are saved:

```json
{
    "save_figures_path": "./custom_figures_folder/",
    "boxplot_path": "./custom_figures_folder/boxplots/"
}
```

## Troubleshooting

### Issue: Plots not displaying
**Solution:** Ensure you're using `plt.show()` for interactive viewing or `plt.savefig()` for saving to disk.

### Issue: Directory not found errors
**Solution:** The code automatically creates directories, but ensure parent directories exist.

### Issue: Memory issues with large datasets
**Solution:** Process data in chunks or use `plt.close()` after each plot to free memory.

## Future Enhancements

Potential additions to the dashboard functionality:
- Interactive dashboards using Plotly or Dash
- Real-time monitoring during model training
- Feature importance visualizations
- Residual analysis plots
- Cross-validation score distributions
- Hyperparameter tuning visualization
- Learning curves
- Confusion matrices for classification variants

## Conclusion

This dashboard system provides a complete visual analytics pipeline for regression analysis. It enables:
- **Data scientists** to understand their data deeply
- **Stakeholders** to see clear performance comparisons
- **Quality assurance** through visual inspection at each step
- **Documentation** of the entire analysis process

All visualizations are automatically saved and can be included in reports, presentations, or documentation.
