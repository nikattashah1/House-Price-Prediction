# ============================================================
# STEP 5: LINEAR REGRESSION FROM SCRATCH (using NumPy only)
# Shows how a model learns using gradient descent, a cost function
# and a learning rate. The result is compared with scikit-learn.
# Run:  python 5_linear_regression_scratch.py
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("LINEAR REGRESSION FROM SCRATCH")
print("-" * 40)

df = pd.read_csv("cleaned_house_data.csv")
os.makedirs("charts", exist_ok=True)

# 1. Prepare the data (same as Step 3)
y = df["total_price"].values / 1000000                    # price in millions
X = pd.get_dummies(df.drop(columns=["total_price"]), columns=["location"], dtype=int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Scale the inputs so gradient descent works well: (value - mean) / std
mean = X_train.mean()
std = X_train.std().replace(0, 1)          # avoid dividing by zero
X_train_s = ((X_train - mean) / std).values.astype(float)
X_test_s = ((X_test - mean) / std).values.astype(float)


# 3. Gradient descent
#    prediction = X x w + b
#    cost       = (1 / 2m) x sum of (prediction - actual)^2      (mean squared error)
#    The weights are moved a small step (learning rate) against the slope of the cost.
def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    cost_history = []
    for i in range(iterations):
        prediction = X @ w + b
        error = prediction - y
        cost = np.sum(error ** 2) / (2 * m)
        cost_history.append(cost)
        w = w - learning_rate * (X.T @ error) / m
        b = b - learning_rate * np.sum(error) / m
    return w, b, cost_history


# 4. Try three learning rates and compare how the cost falls
iterations = 1000
plt.figure(figsize=(8, 5))
for lr in [0.001, 0.01, 0.1]:
    w, b, costs = gradient_descent(X_train_s, y_train, lr, iterations)
    plt.plot(costs, label="learning rate = " + str(lr))
    print("Learning rate", lr, "-> final cost:", round(costs[-1], 4))
plt.title("Cost Function During Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.savefig("charts/8_gradient_descent_cost.png", dpi=150)

# 5. Use the best learning rate (0.1) for the final model and test it
w, b, costs = gradient_descent(X_train_s, y_train, 0.1, iterations)
scratch_r2 = r2_score(y_test, X_test_s @ w + b)

# 6. Compare with scikit-learn's LinearRegression
sk_model = LinearRegression().fit(X_train, y_train)
sk_r2 = r2_score(y_test, sk_model.predict(X_test))

print("\nTest R2 score (from scratch):       ", round(scratch_r2, 4))
print("Test R2 score (scikit-learn):       ", round(sk_r2, 4))
print("Both should be very close to each other.")
plt.show()
