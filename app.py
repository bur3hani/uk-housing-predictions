import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load or download model
# -----------------------------
MODEL_URL = "https://huggingface.co/Bur3hani/UK_Housing_Price_Predictor/resolve/main/uk_property_price_model.pkl"
MODEL_PATH = "uk_property_price_model.pkl"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Hugging Face..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = joblib.load(MODEL_PATH)

# -----------------------------
# UI
# -----------------------------
st.title("🏡 UK Property Price Predictor")
st.markdown("Estimate UK property prices based on location, type, and sale date.")
st.markdown("📥 [Download the dataset used in this model](https://www.kaggle.com/datasets/burhanimtengwa/uk-housing-cleaned)")

property_types = ['D', 'F', 'O', 'S', 'T']
counties = ['GREATER LONDON', 'WEST MIDLANDS', 'GREATER MANCHESTER', 'WEST YORKSHIRE',
            'MERSEYSIDE', 'WEST SUSSEX', 'ESSEX', 'SURREY', 'HAMPSHIRE', 'KENT']

ptype = st.sidebar.selectbox("Property Type", property_types)
county = st.sidebar.selectbox("County", counties)
year = st.sidebar.slider("Year", 1995, 2026, 2025)
month = st.sidebar.slider("Month", 1, 12, 6)

# -----------------------------
# Create Input Data
# -----------------------------
def create_input(property_type, county, year, month):
    input_data = pd.DataFrame(data=0, index=[0], columns=model.feature_names_in_)
    input_data.at[0, "Year"] = year
    input_data.at[0, "Month"] = month
    ptype_col = f"Property_Type_{property_type}"
    county_col = f"County_{county}"
    if ptype_col in input_data.columns:
        input_data.at[0, ptype_col] = 1
    if county_col in input_data.columns:
        input_data.at[0, county_col] = 1
    return input_data

# -----------------------------
# Predict & Display
# -----------------------------
if st.sidebar.button("Predict Price"):
    input_data = create_input(ptype, county, year, month)
    prediction = model.predict(input_data)[0]
    st.subheader(f"💰 Estimated Price: £{prediction:,.0f}")

# -----------------------------
# Future Price Scenarios Table
# -----------------------------
st.markdown("---")
st.markdown("### 🔮 Projected Prices for 2025–2026")

future_years = [2025, 2026]
future_months = [1, 4, 7, 10]
sample_types = ['F', 'S', 'D']
sample_counties = ['GREATER LONDON', 'WEST MIDLANDS', 'SURREY']

records = []
for fy in future_years:
    for fm in future_months:
        for ft in sample_types:
            for fc in sample_counties:
                data = create_input(ft, fc, fy, fm)
                price = model.predict(data)[0]
                records.append({
                    "Year": fy,
                    "Month": fm,
                    "Type": ft,
                    "County": fc,
                    "Price (£)": round(price)
                })

forecast_df = pd.DataFrame(records)
st.dataframe(forecast_df.style.format({"Price (£)": "£{:,}"}))

# -----------------------------
# Optional: Show Exploratory Charts
# -----------------------------
st.markdown("---")
st.markdown("### 📊 Exploratory Data Visualizations")

# Simulated EDA DataFrame (or load a sample)
@st.cache_data
def load_sample_data():
    sample_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/House%20Prices%20Advanced%20Regression%20Techniques/train.csv"
    return pd.read_csv(sample_url)

try:
    sample_df = load_sample_data()
    st.markdown("**Sample Price Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(sample_df['SalePrice'], bins=40, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("**Sale Price by Year Built**")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=sample_df['YearBuilt'], y=sample_df['SalePrice'], ax=ax2)
    ax2.set_xticks(ax2.get_xticks()[::10])
    ax2.set_xticklabels(ax2.get_xticks(), rotation=45)
    st.pyplot(fig2)
except Exception as e:
    st.info("🔍 Sample charts are displayed only when sample dataset loads successfully.")
