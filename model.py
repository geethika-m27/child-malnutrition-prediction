import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load the cleaned dataset
df = pd.read_csv("data/cleaned-malnutrition.csv")

print("=" * 55)
print("GLOBAL CHILD MALNUTRITION PREDICTION")
print("=" * 55)

# STEP 2: Select the target and input features
target = "Stunting"

features = [
    "Country",
    "Year",
    "Income Classification"
]

# Keep only required columns and remove rows without the target
df = df[features + [target]].dropna(subset=[target])

X = df[features]
y = df[target]

print("\nDataset records:", len(df))
print("Input features:", features)
print("Target variable:", target)

# STEP 3: Prepare categorical and numerical features
categorical_features = ["Country"]
numerical_features = ["Year", "Income Classification"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "country",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numbers",
            SimpleImputer(strategy="median"),
            numerical_features
        )
    ]
)

# STEP 4: Build the Random Forest Regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# STEP 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 6: Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# STEP 7: Predict the test data
y_pred = model.predict(X_test)

# STEP 8: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 35)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.3f}")

# STEP 9: Display sample predictions
results = pd.DataFrame({
    "Country": X_test["Country"].values,
    "Actual Stunting Rate (%)": y_test.values,
    "Predicted Stunting Rate (%)": y_pred
})

print("\nSAMPLE PREDICTIONS")
print(results.head(10).round(2).to_string(index=False))

# STEP 10: Save predictions
results.to_csv("data/prediction-results.csv", index=False)

print("\nPredictions saved to data/prediction-results.csv")
print("=" * 55)
