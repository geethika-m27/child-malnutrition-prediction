import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("data/cleaned-malnutrition.csv")

print("DATA ANALYSIS STARTED")
print("Total records:", len(df))

# ----------------------------------
# GRAPH 1: Top 10 countries by stunting
# ----------------------------------

country_stunting = (
    df.groupby("Country")["Stunting"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
country_stunting.sort_values().plot(kind="barh")

plt.title("Top 10 Countries by Average Child Stunting Rate")
plt.xlabel("Stunting Rate (%)")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig("data/top10_stunting.png")
plt.show()

# ----------------------------------
# GRAPH 2: Stunting trends over time
# ----------------------------------

yearly_stunting = (
    df.groupby("Year")["Stunting"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(10, 5))
plt.plot(
    yearly_stunting.index,
    yearly_stunting.values,
    marker="o"
)

plt.title("Average Child Stunting Estimates Over Time")
plt.xlabel("Year")
plt.ylabel("Average Stunting Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/stunting_trend.png")
plt.show()

# ----------------------------------
# GRAPH 3: Correlation heatmap
# ----------------------------------

nutrition_columns = [
    "Severe Wasting",
    "Wasting",
    "Overweight",
    "Stunting",
    "Underweight"
]

correlation = df[nutrition_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Malnutrition Indicators")
plt.tight_layout()
plt.savefig("data/malnutrition_correlation.png")
plt.show()

print("\nAll graphs generated successfully!")
print("Saved in the data folder:")
print("1. top10_stunting.png")
print("2. stunting_trend.png")
print("3. malnutrition_correlation.png")
