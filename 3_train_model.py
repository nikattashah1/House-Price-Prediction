# ============================================================
# STEP 3: TRAIN 3 MODELS AND COMPARE THEIR ACCURACY
# Trains Linear Regression, Decision Tree and Random Forest,
# compares them, and saves all three models.
# Run:  python 3_train_model.py
# ============================================================
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("TRAIN MODEL")
print("-" * 40)

df = pd.read_csv("cleaned_house_data.csv")
os.makedirs("charts", exist_ok=True)

# 1. Separate the inputs (X) from the value we want to predict (y)
y = df["total_price"]                    # target: what we predict
X = df.drop(columns=["total_price"])     # inputs: house details

# Safety check: the cost columns must NOT be used as inputs
for col in ["land_cost", "construction_cost", "material_cost"]:
    assert col not in X.columns, "Data leakage: " + col + " is in the inputs!"

# 2. Convert city names into numbers (one column per city with 0/1)
X = pd.get_dummies(X, columns=["location"], dtype=int)
print("Columns used for training:")
print(list(X.columns))

# 3. Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\nTraining rows:", len(X_train), "| Testing rows:", len(X_test))

# 4. The three models we will compare
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

# 5. Train each model, then measure its accuracy on training data and test data
results = []
best_name = None
best_test_r2 = -999999

for name, model in models.items():
    model.fit(X_train, y_train)                        # the model learns here

    train_r2 = r2_score(y_train, model.predict(X_train))
    predictions = model.predict(X_test)                # predict houses it has never seen
    test_r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append([name, round(train_r2 * 100, 2), round(test_r2 * 100, 2),
                    round(mae), round(rmse)])

    if test_r2 > best_test_r2:                         # remember the best model so far
        best_test_r2 = test_r2
        best_name = name

# 6. Show the ACCURACY COMPARISON table
#    Accuracy (%) here means R2 score x 100 (100% = perfect predictions).
table = pd.DataFrame(results, columns=[
    "Model", "Train Accuracy (%)", "Test Accuracy (%)", "MAE (Rs.)", "RMSE (Rs.)"])
print("\nACCURACY COMPARISON")
print(table.to_string(index=False))
print("\nBest model:", best_name)

# 7. Train every model on all the data and save them for prediction
best_model = models[best_name]
best_predictions = best_model.predict(X_test)     # used for the chart below
for model in models.values():
    model.fit(X, y)                                # train again on ALL the data

joblib.dump({
    "models": models,
    "model": models[best_name],                    # kept for compatibility
    "columns": list(X.columns),
    "name": best_name,
    "area_range": [float(df["land_area_sqft"].min()), float(df["land_area_sqft"].max())],
    "test_accuracy": {
        row[0]: row[2] for row in results
    },
}, "house_price_model.joblib")
print("Saved all 3 models: house_price_model.joblib")

# 8. Chart: accuracy comparison of the three models
plt.figure(figsize=(7, 5))
plt.bar(table["Model"], table["Test Accuracy (%)"],
        color=["steelblue", "orange", "seagreen"], edgecolor="black")
for i, value in enumerate(table["Test Accuracy (%)"]):
    plt.text(i, value, str(value) + "%", ha="center", va="bottom")
plt.title("Model Accuracy Comparison (Test Data)")
plt.ylabel("Accuracy (%)")
plt.grid(axis="y")
plt.savefig("charts/5_model_comparison.png", dpi=150)

# 9. Chart: actual price vs predicted price for the best model
plt.figure(figsize=(6, 6))
plt.scatter(y_test / 1000000, best_predictions / 1000000, edgecolor="black")
line_end = max(y_test.max(), best_predictions.max()) / 1000000
plt.plot([0, line_end], [0, line_end], "r--", label="Perfect prediction")
plt.title("Actual vs Predicted Price (" + best_name + ")")
plt.xlabel("Actual Price (Millions Rs.)")
plt.ylabel("Predicted Price (Millions Rs.)")
plt.legend()
plt.grid(True)
plt.savefig("charts/6_actual_vs_predicted.png", dpi=150)

# 10. Chart: which inputs matter most (from the Random Forest)
forest = models["Random Forest"]
importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
plt.barh(importance.index, importance.values, color="seagreen", edgecolor="black")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("charts/7_feature_importance.png", dpi=150)

print("Saved 3 charts in the 'charts' folder.")
plt.show()
