import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

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

property_types = ['D', 'F', 'O', 'S', 'T']  # example types
counties = ['GREATER LONDON', 'WEST MIDLANDS', 'GREATER MANCHESTER', 'WEST YORKSHIRE',
            'MERSEYSIDE', 'WEST SUSSEX', 'ESSEX', 'SURREY', 'HAMPSHIRE', 'KENT']

ptype = st.sidebar.selectbox("Property Type", property_types)
county = st.sidebar.selectbox("County", counties)
year = st.sidebar.slider("Year", 1995, 2023, 2020)
month = st.sidebar.slider("Month", 1, 12, 6)

# -----------------------------
# Create Input Data
# -----------------------------
input_data = pd.DataFrame(data=0, index=[0], columns=model.feature_names_in_)

input_data.at[0, "Year"] = year
input_data.at[0, "Month"] = month

ptype_col = f"Property_Type_{ptype}"
county_col = f"County_{county}"

if ptype_col in input_data.columns:
    input_data.at[0, ptype_col] = 1

if county_col in input_data.columns:
    input_data.at[0, county_col] = 1

# -----------------------------
# Predict & Display
# -----------------------------
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.subheader(f"💰 Estimated Price: £{prediction:,.0f}")
