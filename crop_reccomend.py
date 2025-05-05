import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('data/Crop_Recommendation.csv')

# Data Overview
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Custom color palette
custom_color_seq = px.colors.qualitative.Dark24  # Clean teal gradient
# Use Dark24 for maximum distinction between crop colors
distinct_colors = px.colors.qualitative.Dark24

bright_colors = ['#1f77b4',  # blue
                 '#ff7f0e',  # orange
                 '#2ca02c',  # green
                 '#d62728',  # red
                 '#ffdd57']  # yellow
custom_color_seq = bright_colors

# Pie Chart: Crop Distribution
pie_fig = px.pie(df, names='Crop', title='Crop Distribution',
                 color_discrete_sequence=custom_color_seq)
pie_fig.update_traces(textinfo='percent+label', pull=[0.05]*len(df['Crop'].unique()))
pie_fig.show()

# Scatter Plot: Rainfall vs Temperature
scatter_fig = px.scatter(df, x='Rainfall', y='Temperature', color='Crop',
                         size='Nitrogen', hover_data=['Humidity', 'pH_Value'],
                         title='Rainfall vs Temperature (Nitrogen Bubble Size)',
                         color_discrete_sequence=custom_color_seq)
scatter_fig.show()

# Box Plot: Nutrients by Crop
box_fig = px.box(df.melt(id_vars='Crop', value_vars=['Nitrogen', 'Phosphorus', 'Potassium']),
                 x='variable', y='value', color='Crop',
                 title='Nutrient Distribution (Nitrogen, Phosphorus, Potassium) by Crop',
                 labels={'variable': 'Nutrient', 'value': 'Level'},
                 color_discrete_sequence=custom_color_seq)
box_fig.show()

# Bar Chart: Average Nutrient Levels by Crop
mean_nutrients = df.groupby('Crop')[['Nitrogen', 'Phosphorus', 'Potassium']].mean().reset_index()
melted = mean_nutrients.melt(id_vars=['Crop'], value_vars=['Nitrogen', 'Phosphorus', 'Potassium'])

bar_nutrients_fig = px.bar(melted, x='Crop', y='value', color='variable',
                           barmode='group', title='Average Nutrient Levels by Crop',
                           labels={'variable': 'Nutrient', 'value': 'Average Level'},
                           color_discrete_sequence=custom_color_seq)
bar_nutrients_fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
bar_nutrients_fig.show()

# Heatmap: Correlation Matrix
corr_matrix = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']].corr()
heatmap_fig = ff.create_annotated_heatmap(z=np.array(corr_matrix),
                                          x=corr_matrix.columns.tolist(),
                                          y=corr_matrix.columns.tolist(),
                                          colorscale='YlGnBu', showscale=True)
heatmap_fig.update_layout(title_text='Correlation Heatmap of Variables')
heatmap_fig.show()

# Sunburst Chart: Nutrients by Crop
sunburst_fig = px.sunburst(df, path=['Crop'], values='Nitrogen',
                           color='Phosphorus', hover_data=['Potassium'],
                           title='Soil Nutrient Levels by Crop',
                           color_continuous_scale='Teal')
sunburst_fig.show()

# Histogram: pH Value Distribution
hist_fig = px.histogram(df, x='pH_Value', color='Crop', nbins=20,
                        title='Distribution of pH Values by Crop',
                        color_discrete_sequence=px.colors.qualitative.Bold)
hist_fig.show()

# 3D Scatter Plot: Temperature, Humidity, Rainfall
scatter_3d_fig = px.scatter_3d(df, x='Temperature', y='Humidity', z='Rainfall', color='Crop',
                               size='Nitrogen', title='Temperature, Humidity, Rainfall by Crop',
                               color_discrete_sequence=custom_color_seq)
scatter_3d_fig.show()

# Model Training: Random Forest
X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value']]
y = df['Crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report Heatmap
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(df_report.iloc[:-1, :].astype(float), annot=True, cmap='YlGnBu', fmt=".2f", vmin=0, vmax=1)
plt.title('Classification Report Heatmap')
plt.xlabel('Metrics')
plt.ylabel('Crops')
plt.tight_layout()
plt.show()
