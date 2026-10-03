# ============================================================
# STEP 2: DATA VISUALIZATION
# Draws charts from the cleaned data and saves them in the "charts" folder
# Run:  python 2_data_visualization.py
# ============================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 40)

df = pd.read_csv("cleaned_house_data.csv")
os.makedirs("charts", exist_ok=True)

# ---- EDA summary printed for the project report ----
print("\nDATASET SHAPE:", df.shape)
print("\nDATA TYPES AND MISSING VALUES:")
print(pd.DataFrame({"dtype": df.dtypes.astype(str),
                    "missing": df.isna().sum(),
                    "unique": df.nunique()}))
print("\nDUPLICATE ROWS:", int(df.duplicated().sum()))
print("\nNUMERICAL SUMMARY:")
print(df.describe().round(2).to_string())
print("\nROWS PER CITY:")
print(df["location"].value_counts().to_string())

q1 = df["total_price"].quantile(0.25)
q3 = df["total_price"].quantile(0.75)
iqr = q3 - q1
price_outliers = df[(df["total_price"] < q1 - 1.5 * iqr) |
                    (df["total_price"] > q3 + 1.5 * iqr)]
print("\nPRICE OUTLIERS (1.5 x IQR):", len(price_outliers))

# Give every city its own color (same colors in all charts)
cities = sorted(df["location"].unique())
colors = dict(zip(cities, sns.color_palette("Set2", len(cities))))
price_in_millions = df["total_price"] / 1000000

# ---- Chart 1: four charts in one figure (subplots) ----
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) Bar chart: average price in each city
avg_price = df.groupby("location")["total_price"].mean() / 1000000
avg_price = avg_price.sort_values()
axes[0, 0].bar(avg_price.index, avg_price.values,
               color=[colors[city] for city in avg_price.index], edgecolor="black")
axes[0, 0].set_title("Average House Price in Each City")
axes[0, 0].set_xlabel("City")
axes[0, 0].set_ylabel("Price (Millions Rs.)")
axes[0, 0].grid(axis="y")

# (b) Scatter plot: land area vs price
sns.scatterplot(data=df, x="land_area_sqft", y=price_in_millions,
                hue="location", palette=colors, s=40, ax=axes[0, 1])
axes[0, 1].set_title("Land Area vs House Price")
axes[0, 1].set_xlabel("Land Area (Sq. Ft.)")
axes[0, 1].set_ylabel("Price (Millions Rs.)")
axes[0, 1].grid(True)

# (c) Histogram: how prices are spread
axes[1, 0].hist(price_in_millions, bins=15, color="steelblue", edgecolor="black")
axes[1, 0].set_title("Distribution of House Prices")
axes[1, 0].set_xlabel("Price (Millions Rs.)")
axes[1, 0].set_ylabel("Number of Houses")

# (d) Box plot: price range in each city
sns.boxplot(data=df, x="location", y=price_in_millions, order=cities, hue="location",
            palette=colors, legend=False, ax=axes[1, 1])
axes[1, 1].set_title("Price Range in Each City")
axes[1, 1].set_xlabel("City")
axes[1, 1].set_ylabel("Price (Millions Rs.)")

fig.tight_layout()
fig.savefig("charts/1_overview_subplots.png", dpi=150)

# ---- Chart 2: Pie chart - share of houses in each city ----
city_counts = df["location"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(city_counts.values, labels=city_counts.index, autopct="%1.1f%%",
        colors=[colors[city] for city in city_counts.index])
plt.title("Share of Houses in Each City")
plt.savefig("charts/2_pie_city_share.png", dpi=150)

# ---- Chart 3: Correlation heatmap (which columns move with price) ----
plt.figure(figsize=(9, 7))
sns.heatmap(df.drop(columns=["location"]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Columns")
plt.tight_layout()
plt.savefig("charts/3_correlation_heatmap.png", dpi=150)

# ---- Chart 4: Pairplot of the selected model features ----
main_columns = ["land_area_sqft", "floors", "bedrooms", "bathrooms", "total_price",
                "location"]
pair = sns.pairplot(df[main_columns], hue="location", palette=colors)
pair.savefig("charts/4_pairplot.png", dpi=100)

print("Saved 4 chart files in the 'charts' folder.")
plt.show()
