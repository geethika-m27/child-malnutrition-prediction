import pandas as pd

# Load the child malnutrition dataset
df = pd.read_csv("data/malnutrition-estimates.csv")

# Display basic dataset information
print("========================================")
print(" CHILD MALNUTRITION DATASET EXPLORATION")
print("========================================")

print("\n1. Dataset loaded successfully!")

print("\n2. Dataset dimensions:")
print("Rows and columns:", df.shape)

print("\n3. Column names:")
print(df.columns.tolist())

print("\n4. First five records:")
print(df.head())

print("\n5. Missing values in each column:")
print(df.isnull().sum())

print("\n6. Data types:")
print(df.dtypes)

print("\n========================================")
print(" DATASET EXPLORATION COMPLETED")
print("========================================")
