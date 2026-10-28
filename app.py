import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="Child Malnutrition Prediction",
    page_icon="🌍",
    layout="centered"
)

# Title and description
st.title("🌍 Global Child Malnutrition Analysis")
st.subheader("Child Stunting Rate Prediction")

st.write(
    "This educational machine learning application estimates "
    "population-level child stunting rates using historical survey data."
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned-malnutrition.csv")

df = load_data()

# Prepare features
features = ["Country", "Year", "Income Classification"]
target = "Stunting"

model_data = df[features + [target]].dropna(subset=[target])

X = model_data[features]
y = model_data[target]

# Train the model
@st.cache_resource
def train_model(X, y):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "country",
                OneHotEncoder(handle_unknown="ignore"),
                ["Country"]
            ),
            (
                "numbers",
                SimpleImputer(strategy="median"),
                ["Year", "Income Classification"]
            )
        ]
    )

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

    model.fit(X, y)
    return model

model = train_model(X, y)

# User input
st.header("Enter Prediction Details")

countries = sorted(model_data["Country"].unique())

country = st.selectbox(
    "Select Country",
    countries
)

year = st.number_input(
    "Enter Year",
    min_value=int(model_data["Year"].min()),
    max_value=int(model_data["Year"].max()) + 5,
    value=int(model_data["Year"].max())
)

country_records = model_data[model_data["Country"] == country]

income = int(
    country_records["Income Classification"].mode().iloc[0]
)

# Prediction
if st.button("Predict Stunting Rate"):
    input_data = pd.DataFrame({
        "Country": [country],
        "Year": [year],
        "Income Classification": [income]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Stunting Rate: {prediction:.2f}%")

    st.info(
        "This is a model-generated estimate, not a medical diagnosis "
        "or an official country statistic."
    )

    st.caption(
        "The model uses historical survey records and may be less reliable "
        "for years or countries with limited data."
    )

# Dataset overview
st.divider()
st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Survey Records", len(model_data))

with col2:
    st.metric("Countries", model_data["Country"].nunique())

st.write("### Average Stunting Rate by Country")

country_average = (
    model_data.groupby("Country")["Stunting"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

st.bar_chart(country_average)

st.caption(
    "Data source: Malnutrition Across the Globe dataset. "
    "Rates summarize available survey records and may cover different years."
)
