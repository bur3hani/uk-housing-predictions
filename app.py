import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model (assumes model was saved via joblib or pickle)
model = joblib.load("property_price_model.joblib")  # replace with actual file path/name

# Get the feature names the model was trained on
# This will include 'Year', 'Month', and all one-hot encoded feature column names
feature_columns = list(model.feature_names_in_)  # ensure it's a Python list

# (Optional) Define or retrieve category options for inputs (should match training categories)
property_type_options = ["Detached", "Semi-Detached", "Terraced", "Flat"]  # example categories
county_options = ["County A", "County B", "County C"]  # example counties (replace with actual)

# Streamlit UI for user inputs
st.title("UK Property Price Prediction")
st.write("Provide property details to predict the price:")

selected_property_type = st.selectbox("Property Type", property_type_options)
selected_county = st.selectbox("County", county_options)
selected_year = st.number_input("Year (e.g. 2023)", min_value=1900, max_value=2100, value=2023)
selected_month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=1)

# When the user clicks the Predict button
if st.button("Predict Price"):
    # Create a DataFrame with one row, containing all model feature columns, initialized to 0
    input_data = pd.DataFrame(data=0, index=[0], columns=feature_columns)
    
    # Set the numeric features from user input
    input_data.at[0, "Year"] = selected_year
    input_data.at[0, "Month"] = selected_month
    
    # Set the one-hot feature corresponding to the selected categories to 1
    input_data.at[0, f"property_type_{selected_property_type}"] = 1
    input_data.at[0, f"county_{selected_county}"] = 1
    
    # (At this point, input_data has all columns model expects, others remain 0)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display the result
    st.subheader("Predicted Price:")
    st.write(f"£{prediction:,.0f}")  # format as currency with no decimals
