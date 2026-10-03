# House Price Prediction (Nepal)

A machine learning project that predicts the price of a house from only 5 inputs:
**city, land area, floors, bedrooms and bathrooms.** Land area can be typed in **Ropani or Aana**.
Three models are trained and compared (Linear Regression, Decision Tree, Random Forest).

## Tools used
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, joblib, Tkinter

## Setup
```
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Put your dataset in this folder as `nepal_house_data.csv` (or `nepal_house_data.xlsx`).
It must contain at least these columns: `location, land_area_sqft, floors, bedrooms,
bathrooms, total_price`. Other columns are ignored (and removed in Step 1).

## How to run
| Step | Command | What it does |
|---|---|---|
| 1 | `python 1_data_cleaning.py` | Cleans the data, removes the cost columns, saves `cleaned_house_data.csv` |
| 2 | `python 2_data_visualization.py` | **EDA**: prints summary tables and saves charts in the `charts` folder |
| 3 | `python 3_train_model.py` | Trains 3 models, prints the accuracy comparison, saves all 3 models |
| 4 | `python 4_predict_price.py` | Asks for a house's details in the terminal and predicts its price |
| 5 | `python 5_linear_regression_scratch.py` | Extra lesson: Linear Regression built by hand (cost function, gradient descent) |
| 6 | `python 6_predict_gui.py` | The same prediction in a window (Tkinter, optional) |

You do **not** run every step each time:
- First time, or after the dataset changes: run Steps 1, 2 and 3 in order.
- Afterwards, to predict a house: only Step 4 or Step 6 (Step 3 saved the trained models).
- Step 5 is optional and does not affect the results.

`CODE_EXPLANATION.txt` explains the whole project and every line of every file in simple language.

## Files created while running
| File / folder | Created by | Used by |
|---|---|---|
| `cleaned_house_data.csv` | Step 1 | Steps 2, 3, 5 |
| `charts/` (PNG images) | Steps 2, 3, 5 | your report |
| `house_price_model.joblib` | Step 3 | Steps 4, 6 |

## Method
1. **Data cleaning** - fix column and city names, force numbers to be numbers, remove missing
   values, duplicates, impossible values and extreme price outliers (IQR method), and keep
   only the 5 inputs plus the price.
2. **Avoiding data leakage** - `total_price` is calculated from `land_cost`,
   `construction_cost` and `material_cost`. If the model saw these columns it would only add
   them up and the accuracy would be fake. They are deleted in Step 1, and Step 3 stops with
   an error if any of them is used as an input. `total_price` stays in the cleaned file only
   as the **target** (the answer the model learns to predict); it is never an input.
3. **EDA (Exploratory Data Analysis)** - Step 2 prints the data shape, data types, missing
   values, duplicates, summary statistics, houses per city and the number of price outliers,
   and saves a bar chart, scatter plot, histogram, box plot, pie chart, correlation heatmap
   and pairplot.
4. **Preparing inputs** - city names are converted to 0/1 columns (one-hot encoding).
5. **Training** - data is split 80% training / 20% testing. Linear Regression, Decision Tree
   and Random Forest are trained.
6. **Accuracy comparison** - the table shows train accuracy, test accuracy (R2 score x 100),
   MAE and RMSE for each model. The model with the best test accuracy is the best model.
   Afterwards all 3 models are retrained on all the data and saved.
7. **Prediction** - the user gives the 5 inputs; the program converts the land area to square
   feet (1 Ropani = 5476 sq. ft., 1 Aana = 342.25 sq. ft.) and shows the price from all 3
   models, the most accurate model, and the same house in every city.
8. **Safety warning** - Step 3 saves the smallest and largest land area in the data. If a
   user enters a plot outside that range, Steps 4 and 6 print a warning that the price may
   be unreliable.

## Metrics
- **Accuracy (%)** = R2 score x 100. 100% = perfect, 0% = no better than guessing the average.
- **Train vs test accuracy**: a big gap means overfitting (the model memorised the data).
- **MAE**: average error in rupees. **RMSE**: like MAE but punishes big errors more.
- For MAE and RMSE, smaller is better.

## Limitations
- The models only know the cities present in the dataset.
- The models are only reliable for land sizes within the range of the dataset. Decision Tree and
  Random Forest give the same price for every plot bigger than the largest one they saw, so
  for example 1 Ropani (5,476 sq. ft.) is outside a dataset whose largest plot is 3,000 sq. ft.
- If the dataset was generated with fixed formulas, high accuracy shows the models learned
  those formulas, not real market behaviour. Real listing data would give lower scores.
- Only Ropani and Aana are supported for land units.
- A tiny dataset (for example 13 rows) gives meaningless, even negative, accuracy.

## Future improvements
Use real listing data with larger plots, add more cities and features such as road access and
distance to the city centre, support more land units (bigha, kattha, square feet), and publish
the app as a website.
