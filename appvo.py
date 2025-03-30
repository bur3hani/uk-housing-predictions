# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("uk_property_price_model.pkl")

# Title
st.title("🏡 UK Property Price Predictor")
st.markdown("Estimate property prices in the UK based on location, type, and transaction date.")

# Sidebar inputs
st.sidebar.header("Enter Property Details")

# Load sample feature options from training metadata (manually defined for now)
property_types = ['D', 'F', 'O', 'S', 'T']  # Detached, Flat, Other, Semi, Terraced
counties = [
    'GREATER LONDON', 'WEST MIDLANDS', 'GREATER MANCHESTER', 'WEST YORKSHIRE',
    'MERSEYSIDE', 'WEST SUSSEX', 'ESSEX', 'SURREY', 'HAMPSHIRE', 'KENT'  # etc.
]

# User inputs
property_type = st.sidebar.selectbox("Property Type", property_types)
county = st.sidebar.selectbox("County", counties)
year = st.sidebar.slider("Year of Sale", 1995, 2023, 2020)
month = st.sidebar.slider("Month of Sale", 1, 12, 6)

# Prepare input features
input_data = pd.DataFrame({
    'Year': [year],
    'Month': [month],
})

# Add one-hot encoding for property type and county
for col in model.feature_names_in_:
    if col.startswith("Property_Type_"):
        input_data[col] = 1 if col == f"Property_Type_{property_type}" else 0
    elif col.startswith("County_"):
        input_data[col] = 1 if col == f"County_{county}" else 0

# Ensure all model features are present
for col in model.feature_names_in_:
    if col not in input_data:
        input_data[col] = 0

# Predict
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.subheader(f"💰 Estimated Price: £{prediction:,.0f}")
