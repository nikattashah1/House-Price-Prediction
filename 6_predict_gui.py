# ============================================================
# STEP 6: PREDICTION WINDOW (Tkinter GUI)
# The same prediction as Step 4, but in a window with a button.
# Run:  python 6_predict_gui.py
# ============================================================
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

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
cities = [c.replace("location_", "") for c in columns if c.startswith("location_")]
LAND_UNITS_SQFT = {"Ropani": 5476.0, "Aana": 342.25}


def predict_prices(house_details):
    """Return the predicted price from every trained model."""
    row = pd.DataFrame([house_details])
    row = pd.get_dummies(row, columns=["location"], dtype=int)
    row = row.reindex(columns=columns, fill_value=0)
    return {name: model.predict(row)[0] for name, model in models.items()}


# 2. Button action: read the boxes, predict and show the result
def on_predict():
    try:
        unit = unit_box.get()
        if unit not in LAND_UNITS_SQFT:
            raise ValueError("Please choose Ropani or Aana.")
        try:
            land_amount = float(land_amount_entry.get())
        except ValueError:
            raise ValueError("Land area must be a number.")
        if land_amount <= 0:
            raise ValueError("Land area must be above 0.")

        house = {
            "location": city_box.get(),
            "land_area_sqft": land_amount * LAND_UNITS_SQFT[unit],
        }
        if house["location"] not in cities:
            raise ValueError("Please choose a city from the list.")

        for key in number_fields:
            try:
                house[key] = float(entries[key].get())
            except ValueError:
                raise ValueError(number_fields[key] + " must be a number.")
            if house[key] < 0:
                raise ValueError(number_fields[key] + " cannot be negative.")
        if house["floors"] < 1:
            raise ValueError("Floors must be at least 1.")

        prices = predict_prices(house)
        result_lines = ["Predicted prices:"]
        result_lines.extend(
            "  {:<18} Rs. {:,.0f}".format(name, price)
            for name, price in prices.items()
        )
        if test_accuracy:
            accurate_name = max(test_accuracy, key=test_accuracy.get)
            result_lines.append(
                "\nMost accurate model: {} ({:.2f}% test accuracy)".format(
                    accurate_name, test_accuracy[accurate_name])
            )
        area_range = saved.get("area_range")
        if area_range and not (area_range[0] <= house["land_area_sqft"] <= area_range[1]):
            result_lines.append("\nWARNING: land size is outside {:,.0f} - {:,.0f} sq. ft.".format(
                area_range[0], area_range[1]))
            result_lines.append("(prices may be unreliable)")
        result_label.config(text="\n".join(result_lines))

        lines = []
        for c in cities:
            other = dict(house)
            other["location"] = c
            best_price = predict_prices(other)[saved["name"]]
            lines.append("{:<10} Rs. {:>14,.0f}".format(c, best_price))
        compare_label.config(
            text="Same house in each city (" + saved["name"] + "):\n" + "\n".join(lines))

    except ValueError as error:
        messagebox.showerror("Invalid input", "Please check your values.\n\n" + str(error))


def on_clear():
    land_amount_entry.delete(0, tk.END)
    for entry in entries.values():
        entry.delete(0, tk.END)
    result_label.config(text="")
    compare_label.config(text="")


# 3. Build the window
window = tk.Tk()
window.title("House Price Prediction")
window.resizable(False, False)

tk.Label(window, text="House Price Prediction", font=("Arial", 16, "bold")).grid(
    row=0, column=0, columnspan=2, pady=10)

tk.Label(window, text="City").grid(row=1, column=0, sticky="e", padx=10, pady=4)
city_box = ttk.Combobox(window, values=cities, state="readonly", width=18)
city_box.grid(row=1, column=1, padx=10, pady=4)
city_box.current(0)

tk.Label(window, text="Land area").grid(row=2, column=0, sticky="e", padx=10, pady=4)
land_amount_entry = tk.Entry(window, width=21)
land_amount_entry.grid(row=2, column=1, padx=10, pady=4)
unit_box = ttk.Combobox(window, values=list(LAND_UNITS_SQFT), state="readonly", width=18)
unit_box.grid(row=3, column=1, padx=10, pady=4)
unit_box.current(0)
tk.Label(window, text="Land unit").grid(row=3, column=0, sticky="e", padx=10, pady=4)

number_fields = {
    "floors": "Floors",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
}
entries = {}
row_number = 4
for key, text in number_fields.items():
    tk.Label(window, text=text).grid(row=row_number, column=0, sticky="e", padx=10, pady=4)
    entries[key] = tk.Entry(window, width=21)
    entries[key].grid(row=row_number, column=1, padx=10, pady=4)
    row_number += 1

tk.Button(window, text="Predict Price", command=on_predict, bg="#2a9d8f",
          fg="white", width=14).grid(row=row_number, column=0, pady=12)
tk.Button(window, text="Clear", command=on_clear, width=14).grid(
    row=row_number, column=1, pady=12)

result_label = tk.Label(window, text="", font=("Arial", 13, "bold"), fg="darkgreen")
result_label.grid(row=row_number + 1, column=0, columnspan=2)
compare_label = tk.Label(window, text="", font=("Courier", 10), justify="left")
compare_label.grid(row=row_number + 2, column=0, columnspan=2, pady=8)

window.mainloop()
