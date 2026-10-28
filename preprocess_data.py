import pandas as pd

# Step 1: Load the original dataset
df = pd.read_csv("data/malnutrition-estimates.csv")

print("STEP 1: Original dataset")
print("Shape:", df.shape)

# Step 2: Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Step 3: Convert sample size to numeric where possible
df["Survey Sample (N)"] = pd.to_numeric(
    df["Survey Sample (N)"], errors="coerce"
)

# Step 4: Convert nutrition indicators to numeric
nutrition_columns = [
    "Severe Wasting",
    "Wasting",
    "Overweight",
    "Stunting",
    "Underweight"
]

for column in nutrition_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Step 5: Check missing values
print("\nSTEP 2: Missing values before cleaning")
print(df[nutrition_columns].isnull().sum())

# Step 6: Remove rows without the target value
df = df.dropna(subset=["Stunting"]).copy()

# Step 7: Remove duplicate rows
df = df.drop_duplicates()

# Step 8: Save the cleaned dataset
df.to_csv("data/cleaned-malnutrition.csv", index=False)

print("\nSTEP 3: Cleaning completed")
print("Cleaned dataset shape:", df.shape)
print("Missing Stunting values:", df["Stunting"].isnull().sum())
print("\nCleaned dataset saved to:")
print("data/cleaned-malnutrition.csv")

print("\nFirst five cleaned records:")
print(df.head())