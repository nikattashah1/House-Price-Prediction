# STEP 4: PREDICT THE PRICE OF A NEW HOUSE (text version)
# Asks for the house details and prints all three predicted prices.
# Run:  python 4_predict_price.py

import joblib
import pandas as pd

print("HOUSE PRICE PREDICTION")
print("-" * 40)

# 1. Load the model saved by 3_train_model.py
try:
    saved = joblib.load("house_price_model.joblib")
except FileNotFoundError:
    raise SystemExit("Model not found. Run 3_train_model.py first.")

models = saved.get("models")
if not models:
    raise SystemExit(
        "This model file contains only the best model. "
        "Run 3_train_model.py again to save all 3 models."
    )
columns = saved["columns"]
test_accuracy = saved.get("test_accuracy", {})

# Cities the model knows (taken from its "location_..." columns)
cities = [c.replace("location_", "") for c in columns if c.startswith("location_")]
LAND_UNITS_SQFT = {"Ropani": 5476.0, "Aana": 342.25}
print("Best model from comparison:", saved["name"])
print("Cities available:", cities)
print("Land units: 1 Ropani = 5476 sq. ft.; 1 Aana = 342.25 sq. ft.")
print()


def ask_number(question, minimum=0, maximum=None):
    """Keeps asking until the user types a valid number."""
    while True:
        try:
            value = float(input(question))
        except ValueError:
            print("  Invalid input. Please type a number.")
            continue
        if value < minimum or (maximum is not None and value > maximum):
            print("  Value out of range. Try again.")
            continue
        return value


def ask_land_area():
    """Ask for a Nepal land unit and convert it to square feet."""
    while True:
        unit = input("Land unit (Ropani/Aana): ").strip().title()
        if unit in LAND_UNITS_SQFT:
            break
        print("  Please enter Ropani or Aana.")
    amount = ask_number("Enter land area in {}: ".format(unit), minimum=0.01)
    area_in_sqft = amount * LAND_UNITS_SQFT[unit]
    print("  That is {:,.1f} sq. ft.".format(area_in_sqft))

    # The models can only predict well for land sizes similar to the ones they learned from
    area_range = saved.get("area_range")
    if area_range and (area_in_sqft < area_range[0] or area_in_sqft > area_range[1]):
        print("  WARNING: the models learned from land between {:,.0f} and {:,.0f} sq. ft.".format(
            area_range[0], area_range[1]))
        print("  This area is outside that range, so the prices below may be unreliable.")
    return area_in_sqft


def prepare_row(house_details):
    """Prepare one house using the same steps as training."""
    row = pd.DataFrame([house_details])
    row = pd.get_dummies(row, columns=["location"], dtype=int)
    row = row.reindex(columns=columns, fill_value=0)     # same columns as training
    return row


def predict_prices(house_details):
    """Return the predicted price from every trained model."""
    row = prepare_row(house_details)
    return {name: model.predict(row)[0] for name, model in models.items()}


# 2. Ask the user for the house details
while True:
    city = input("Enter city: ").strip().title()
    if city in cities:
        break
    print("  City not in the dataset. Choose from:", cities)

house = {
    "location": city,
    "land_area_sqft": ask_land_area(),
    "floors": ask_number("Enter number of floors: ", minimum=1),
    "bedrooms": ask_number("Enter number of bedrooms: "),
    "bathrooms": ask_number("Enter number of bathrooms: "),
}

# 3. Predict using all three models
print("\nPredicted prices:")
for name, price in predict_prices(house).items():
    print("  {:<18} Rs. {:,.0f}".format(name, price))
if test_accuracy:
    accurate_name = max(test_accuracy, key=test_accuracy.get)
    print("\nMost accurate model: {} (test accuracy: {:.2f}%)".format(
        accurate_name, test_accuracy[accurate_name]))

# 4. Show the same house in every city (to see how location changes the price)
print("\nSame house in each city:")
for c in cities:
    same_house = dict(house)
    same_house["location"] = c
    prices = predict_prices(same_house)
    print("  " + c)
    for name, price in prices.items():
        print("    {:<18} Rs. {:>14,.0f}".format(name, price))
