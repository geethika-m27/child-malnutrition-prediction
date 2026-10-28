import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("data/cleaned-malnutrition.csv")

features = ["Country", "Year", "Income Classification"]
target = "Stunting"

df = df[features + [target]].dropna(subset=[target])

X = df[features]
y = df[target]

# 2. Prepare features
preprocessor = ColumnTransformer([
    ("country", OneHotEncoder(handle_unknown="ignore"), ["Country"]),
    ("numbers", SimpleImputer(strategy="median"),
     ["Year", "Income Classification"])
])

# 3. Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Display evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MODEL EVALUATION")
print("------------------------------")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.3f}")

# 7. Plot actual vs predicted values
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

# Reference line for perfect predictions
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Stunting Rate (%)")
plt.ylabel("Predicted Stunting Rate (%)")
plt.title("Actual vs Predicted Child Stunting Rates")
plt.tight_layout()

# Save graph
plt.savefig("data/actual_vs_predicted.png", dpi=300)
plt.show()

print("\nGraph saved as data/actual_vs_predicted.png")

# 8. Save evaluation results
results = pd.DataFrame({
    "Actual Stunting Rate (%)": y_test.values,
    "Predicted Stunting Rate (%)": y_pred
})

results.to_csv("data/evaluation-results.csv", index=False)

print("Evaluation results saved successfully!")