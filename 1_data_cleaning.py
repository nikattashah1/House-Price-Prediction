# STEP 1: DATA CLEANING
# Reads the raw dataset, cleans it and saves cleaned_house_data.csv
# Run:  python 1_data_cleaning.py

import os
import pandas as pd

print("DATA CLEANING")
print("-" * 40)

# 1. Load the dataset (Excel file or CSV file)
try:
    if os.path.exists("nepal_house_data.xlsx"):
        df = pd.read_excel("nepal_house_data.xlsx")
    elif os.path.exists("nepal_house_data.csv"):
        df = pd.read_csv("nepal_house_data.csv")
    else:
        raise FileNotFoundError("Put nepal_house_data.csv (or .xlsx) in this folder.")
except Exception as error:
    raise SystemExit("Could not load the dataset: " + str(error))

print("Rows before cleaning:", len(df))

# 2. Tidy the column names and city names
df.columns = df.columns.str.strip().str.lower()

required = ["location", "land_area_sqft", "floors", "bedrooms", "bathrooms", "total_price"]
missing_columns = [col for col in required if col not in df.columns]
if missing_columns:
    raise SystemExit("These columns are missing from the dataset: " + str(missing_columns))

df["location"] = df["location"].astype(str).str.strip().str.title()

# 3. Make sure all number columns really contain numbers
#    (wrong text such as "abc" becomes empty and is removed in the next step)
for col in df.columns:
    if col != "location":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Remove rows with missing values and duplicate rows
print("Missing values found:", int(df.isnull().sum().sum()))
df = df.dropna()
df = df.drop_duplicates()

# 5. Remove rows with impossible values
df = df[(df["land_area_sqft"] > 0) & (df["floors"] >= 1) & (df["total_price"] > 0)]

# 6. Handle outliers in the price using the IQR method
#    A price far outside the normal range (more than 3 x IQR away) is removed.
#    This is skipped for very small datasets, where every row matters.
if len(df) >= 30:
    q1 = df["total_price"].quantile(0.25)
    q3 = df["total_price"].quantile(0.75)
    iqr = q3 - q1
    lowest = q1 - 3 * iqr
    highest = q3 + 3 * iqr
    rows_before = len(df)
    df = df[(df["total_price"] >= lowest) & (df["total_price"] <= highest)]
    print("Outliers removed:", rows_before - len(df))
else:
    print("Outlier removal skipped (dataset has fewer than 30 rows)")

# 7. IMPORTANT - REMOVE THE COST COLUMNS (to avoid data leakage)
#    total_price = land_cost + construction_cost + material_cost + small extras.
#    If we keep these columns, the model just adds them up instead of learning
#    from the house details, so its accuracy would be fake.
cost_columns = ["land_cost", "construction_cost", "material_cost"]
df = df.drop(columns=cost_columns, errors="ignore")

# 8. Keep only the simple, user-provided features selected for the model.
model_columns = ["location", "land_area_sqft", "floors", "bedrooms", "bathrooms",
                 "total_price"]
df = df[model_columns]

# 9. Save the cleaned data
df.to_csv("cleaned_house_data.csv", index=False)

print("Rows after cleaning:", len(df))
print("Columns saved:", list(df.columns))
print("\nRows per city:")
print(df["location"].value_counts())
print("\nSaved: cleaned_house_data.csv")
