# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import os
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset
df = pd.read_csv('data/crop_yield.csv')
df.head()
df.tail()
print("Shape of the dataset : ",df.shape)
df.isnull().sum()
df.info()

# Check the duplicates record
df.duplicated().sum()
df.describe()

# Create a folder to save the plots if it doesn't exist
output_folder = 'plots_indian'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Plotting with Regression Line for Yield vs Annual Rainfall and saving the plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x = df['Annual_Rainfall'], y = df['Yield'])
slope, intercept, r_value, p_value, std_err = linregress(df['Annual_Rainfall'], df['Yield'])
plt.plot(df['Annual_Rainfall'], slope * df['Annual_Rainfall'] + intercept, color='red')
plt.xlabel('Annual Rainfall')
plt.ylabel('Yield')
plt.title('Annual Rainfall vs Yield with Regression Line')
plt.legend()
plt.savefig(os.path.join(output_folder, 'Rainfall_vs_Yield.png'))
plt.close()

# Filter data for Crop Year 2020
df_year = df[df['Crop_Year'] != 2020]  # As the data of 2020 is incomplete
year_yield = df_year.groupby('Crop_Year').sum()

# Plotting Yield over the Year with Regression Line and saving the plot
plt.figure(figsize=(12, 5))
sns.lineplot(x=year_yield.index, y=year_yield['Yield'], color='blue', linestyle='dashed', marker='o', markersize=8, markerfacecolor='black')
slope, intercept, r_value, p_value, std_err = linregress(year_yield.index, year_yield['Yield'])
print(f"yield: {slope}")
plt.plot(year_yield.index, slope * year_yield.index + intercept, color='red')
plt.xlabel('Year')
plt.ylabel('Yield')
plt.title('Measure of Yield over the Year with Regression Line')
plt.legend()
plt.savefig(os.path.join(output_folder, 'Yield_over_Year.png'))
plt.close()

# Plotting Area under cultivation over the Year with Regression Line and saving the plot
plt.figure(figsize=(12, 3))
sns.lineplot(x=year_yield.index, y=year_yield['Area'], color='blue', linestyle='dashed', marker='o', markersize=8, markerfacecolor='black')
slope, intercept, r_value, p_value, std_err = linregress(year_yield.index, year_yield['Area'])
print(f"Area under cultivation: {slope}")
plt.plot(year_yield.index, slope * year_yield.index + intercept, color='red')
plt.xlabel('Year')
plt.ylabel('Area')
plt.title('Area under cultivation over the Year with Regression Line')
plt.legend()
plt.savefig(os.path.join(output_folder, 'Area_over_Year.png'))
plt.close()

# Plotting Fertilizer Use over the Year with Regression Line and saving the plot
plt.figure(figsize=(12, 3))
sns.lineplot(x=year_yield.index, y=year_yield['Fertilizer'], color='blue', linestyle='dashed', marker='o', markersize=8, markerfacecolor='black')
slope, intercept, r_value, p_value, std_err = linregress(year_yield.index, year_yield['Fertilizer'])
print(f"fertilizer: {slope}")
plt.plot(year_yield.index, slope * year_yield.index + intercept, color='red')
plt.xlabel('Year')
plt.ylabel('Fertilizer')
plt.title('Use of Fertilizer over the Year with Regression Line')
plt.legend()
plt.savefig(os.path.join(output_folder, 'Fertilizer_over_Year.png'))
plt.close()

# Plotting Pesticide Use over the Year with Regression Line and saving the plot
plt.figure(figsize=(12, 3))
sns.lineplot(x=year_yield.index, y=year_yield['Pesticide'], color='blue', linestyle='dashed', marker='o', markersize=8, markerfacecolor='black')
slope, intercept, r_value, p_value, std_err = linregress(year_yield.index, year_yield['Pesticide'])
print(f"pesticide: {slope}")
plt.plot(year_yield.index, slope * year_yield.index + intercept, color='red')
plt.xlabel('Year')
plt.ylabel('Pesticide')
plt.title('Use of Pesticide over the Year with Regression Line')
plt.legend()
plt.savefig(os.path.join(output_folder, 'Pesticide_over_Year.png'))
plt.close()