import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request
import numpy as np

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
