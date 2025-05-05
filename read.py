import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv("engineered_data_preview.csv")

# Use correct casing for relevant columns
target_columns = ['log1p_production']
correlation_columns = ['fertilizer', 'pesticide', 'area']

# Step 1: Remove rows with missing or zero/negative area or production
df = df.dropna(subset=target_columns + correlation_columns)
df = df[(df['area'] > 0) & (df['log1p_production'] > 0)]

# Step 2: Apply log1p transform to reduce skewness
for col in target_columns + correlation_columns:
    df[f"log1p_{col}"] = np.log1p(df[col])

# Step 3: Compute correlation matrix using transformed columns
#log_target_cols = [f"log1p_{col}" for col in target_columns]
log_target_cols = target_columns.copy()
log_corr_cols = [f"log1p_{col}" for col in correlation_columns]

correlations = df[log_corr_cols + log_target_cols].corr().loc[log_corr_cols, log_target_cols]

# Step 4: Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap (log1p transformed)")
plt.tight_layout()
plt.savefig("correlation_matrix_heatmap.png")
plt.close()
