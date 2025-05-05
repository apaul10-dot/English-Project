import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a directory to save plots
os.makedirs("plots", exist_ok=True)

# Load the dataset
Data = pd.read_csv("data/FAO.csv")

# Drop duplicates
print("Number of duplicates:", Data.duplicated().sum())
Data = Data.drop_duplicates()

# Display head and count missing values
print(Data.head())
print(Data.isna().sum())

# Count the number of data points for each country
country_data_counts = Data['country'].value_counts()

# Print out the result
print(country_data_counts)

# Filter the dataset for India
india_data = Data[Data['country'] == "India"]

# Get all unique food commodities for India
india_commodities = india_data['commodity'].unique()

# Print the list of unique food commodities
print(india_commodities)

# Get all counts commodities for India
india_counts = india_data['commodity'].value_counts()

# Print the list of counts food commodities
print(india_counts)

# Filter for specific countries and calculate mean loss percentage
selected_countries = [
    "Africa", "Asia", "Central Asia", "Europe", "Latin America and the Caribbean",
    "Northern Africa", "Northern America", "South-Eastern Asia", "Southern Asia",
    "Sub-Saharan Africa", "Western Africa", "Western Asia", "Australia and New Zealand", "World"
]

mean_loss = (
    Data[Data['country'].isin(selected_countries)]
    .groupby('country')['loss_percentage']
    .mean()
    .reset_index()
    .sort_values(by="loss_percentage", ascending=False)
)


# Filter dataset for selected regions
region_data = Data[Data['country'].isin(selected_countries)].copy()

# Group by updated region list and calculate mean loss percentage
region_loss = (
    region_data
    .groupby('country')['loss_percentage']
    .mean()
    .reset_index()
    .sort_values(by="loss_percentage", ascending=False)
)

# Remove Australia and New Zealand *only from region_loss*
region_loss = region_loss[region_loss['country'] != "Australia and New Zealand"]

# Sort by loss_percentage for nice plotting
region_loss = region_loss.sort_values(by="loss_percentage", ascending=False)

# Plotting
plt.figure(figsize=(10, 8))

# Set a clean color palette
colors = plt.get_cmap('tab20c').colors

# Pie chart
patches, texts, autotexts = plt.pie(
    region_loss['loss_percentage'],
    labels=region_loss['country'],
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12},
    pctdistance=0.8,
    labeldistance=1.1,
    shadow=False
)

# Make labels bold
for text in texts:
    text.set_fontweight('bold')

# Make percentage text slightly smaller and clean
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

# Title
plt.title("Average Food Loss Percentage by Region", fontsize=16, weight='bold')

# Layout adjustments
plt.tight_layout()

# Save figure
plt.savefig("plots/clean_region_loss_pie_chart.png", dpi=300)
#plt.show()

print(region_loss)

'''
# Group by individual countries (not regions) and calculate mean loss percentage
country_loss = (
    Data.groupby('country')['loss_percentage']
    .mean()
    .reset_index()
    .sort_values(by="loss_percentage", ascending=False)
)

# Optional: If you want to filter out countries with very few samples
# For example, only include countries with at least 5 entries:
# country_counts = Data['country'].value_counts()
# country_loss = country_loss[country_loss['country'].isin(country_counts[country_counts >= 5].index)]

# Make a pie chart
plt.figure(figsize=(12, 12))
plt.pie(
    country_loss['loss_percentage'],
    labels=country_loss['country'],
    autopct='%1.1f%%',
    startangle=140
)
plt.title("Average Food Loss Percentage by Country")
plt.tight_layout()
plt.savefig("plots/country_loss_pie_chart.png")
plt.show()'''


'''
# Boxplot of loss percentage by country
plt.figure(figsize=(10, 6))
sns.boxplot(data=Data[Data['country'].isin(selected_countries)], x="loss_percentage", y="country")
plt.title("Loss Percentage by Country")
plt.savefig("plots/loss_percentage_boxplot.png")
plt.close()

# Filter for specific conditions and sort by loss percentage
filtered_data = (
    Data[
        (Data['country'].isin(selected_countries)) &
        (Data['loss_percentage'] >= 20)
    ]
    .sort_values(by="loss_percentage", ascending=False)
    .loc[:, ["loss_percentage", "loss_percentage_original", "country", "commodity", "year", "food_supply_stage", "activity"]]
)
print(filtered_data)

# Mean loss percentage for countries not in the selected list
other_countries_mean_loss = (
    Data[~Data['country'].isin(selected_countries)]
    .groupby('country')['loss_percentage']
    .mean()
    .reset_index()
    .sort_values(by="loss_percentage", ascending=False)
)
print(other_countries_mean_loss)

# Filter for other countries with loss percentage >= 20
other_countries_filtered = (
    Data[
        (~Data['country'].isin(selected_countries)) &
        (Data['loss_percentage'] >= 20)
    ]
    .sort_values(by="loss_percentage", ascending=False)
    .loc[:, ["country", "commodity", "loss_percentage", "loss_percentage_original", "year", "food_supply_stage", "activity"]]
)
print(other_countries_filtered)

# Bar plot for food supply stage by country
n = 2
for country in selected_countries:
    subset = Data[(Data['country'] == country) & (Data['food_supply_stage'] != "")]
    if subset.empty:
        continue

    stage_mean_loss = (
        subset.groupby('food_supply_stage')['loss_percentage']
        .mean()
        .reset_index()
        .sort_values(by="loss_percentage", ascending=False)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=stage_mean_loss, x="loss_percentage", y="food_supply_stage", palette="viridis")
    plt.title(f"Food Loss Percentage by Supply Stage in {country}")
    for index, row in stage_mean_loss.iterrows():
        plt.text(row['loss_percentage'], index, round(row['loss_percentage'], 2), va='center')
    plt.savefig(f"plots/food_loss_{country}.png")
    plt.close()
    n += 2

# Boxplot of loss percentage by food supply stage
plt.figure(figsize=(10, 6))
sns.boxplot(data=Data, x="loss_percentage", y="food_supply_stage")
plt.title("Loss Percentage by Food Supply Stage")
plt.savefig("plots/loss_percentage_food_stage_boxplot.png")
plt.close()

# Mean loss percentage by food supply stage
stage_mean_loss = (
    Data[Data['food_supply_stage'] != ""]
    .groupby('food_supply_stage')['loss_percentage']
    .mean()
    .reset_index()
    .sort_values(by="loss_percentage", ascending=False)
)
print(stage_mean_loss)

# Filter for Post-harvest stage with loss percentage >= 20
post_harvest_data = (
    Data[
        (~Data['country'].isin(selected_countries)) &
        (Data['loss_percentage'] >= 20) &
        (Data['food_supply_stage'] == "Post-harvest")
    ]
    .sort_values(by="loss_percentage", ascending=False)
    .loc[:, ["country", "commodity", "loss_percentage", "loss_percentage_original", "year", "food_supply_stage", "activity"]]
)
print(post_harvest_data)
'''

from scipy.stats import linregress

# Filter for India and the selected commodity
commodity_name = "Mangoes, guavas and mangosteens"
india_commodity_data = Data[(Data['country'] == "India") & 
                            (Data['commodity'] == commodity_name)]

# Group by year and calculate the average loss percentage
india_yearly_mean = india_commodity_data.groupby('year', as_index=False)['loss_percentage'].mean()

# Calculate linear regression line
slope, intercept, r_value, p_value, std_err = linregress(
    india_yearly_mean['year'], india_yearly_mean['loss_percentage'])

print(f"Slope of the regression line: {slope:.4f}")
print(f"R-squared: {r_value**2:.4f}")

# Plot the time series with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=india_yearly_mean, x="year", y="loss_percentage", 
            scatter_kws={'s': 60}, line_kws={'color': 'red'}, ci=None)

# Add labels and title
plt.title(f"Post-Harvest Losses Over Time in India ({commodity_name})", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Loss Percentage (%)", fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save and close the plot
plt.savefig("plots/india_mangoes_loss_time_series.png")
plt.close()