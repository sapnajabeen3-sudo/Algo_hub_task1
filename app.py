"""
app.py
-------
This is the Streamlit web application for the
Bengaluru House Price Prediction project.

The user selects house details from the sidebar, clicks
"Predict Price", and the app displays the estimated price
using our trained Linear Regression model.
"""

import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------
# Page configuration (must be the first Streamlit command)
# -----------------------------------------------------
st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.75), rgba(255, 255, 255, 0.75)),
                          url("https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.88);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------
# Load the trained model, model columns, and cleaned data
# We cache these so they are loaded only once, making the
# app faster.
# -----------------------------------------------------
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns


@st.cache_data
def load_reference_data():
    # Used only to populate dropdown options (area types & locations)
    df = pd.read_csv("cleaned_data.csv")
    return df


model, model_columns = load_model_and_columns()
data = load_reference_data()

# Get sorted unique values for dropdowns
area_types = sorted(data['area_type'].unique())
locations = sorted(data['location'].unique())

# -----------------------------------------------------
# App Title
# -----------------------------------------------------
st.title("🏠 Bengaluru House Price Prediction")
st.write(
    "Fill in the house details in the sidebar and click "
    "**Predict Price** to estimate the price in Lakhs (INR)."
)

# -----------------------------------------------------
# Sidebar - User Inputs
# -----------------------------------------------------
st.sidebar.header("Enter House Details")

area_type = st.sidebar.selectbox("Area Type", area_types)

location = st.sidebar.selectbox("Location", locations)

bhk = st.sidebar.number_input(
    "BHK (Number of Bedrooms)", min_value=1, max_value=16, value=2, step=1
)

total_sqft = st.sidebar.number_input(
    "Total Square Feet", min_value=300, max_value=30000, value=1200, step=50
)

bath = st.sidebar.number_input(
    "Bathrooms", min_value=1, max_value=16, value=2, step=1
)

balcony = st.sidebar.number_input(
    "Balcony", min_value=0, max_value=5, value=1, step=1
)

predict_button = st.sidebar.button("Predict Price")

# -----------------------------------------------------
# Prediction Logic
# -----------------------------------------------------
if predict_button:

    # Step 1: Create a single-row DataFrame with the raw inputs.
    # Column names here match the columns used before One-Hot Encoding.
    input_df = pd.DataFrame({
        'area_type': [area_type],
        'location': [location],
        'total_sqft': [total_sqft],
        'bath': [bath],
        'balcony': [balcony],
        'bhk': [bhk]
    })

    # Step 2: One-Hot Encode the input the SAME way we did during training
    input_encoded = pd.get_dummies(
        input_df, columns=['area_type', 'location'], drop_first=True
    )

    # Step 3: Make sure the input has exactly the same columns (and order)
    # as the columns the model was trained on.
    # Any column missing in the input is added with a value of 0.
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Step 4: Predict the price using the trained Linear Regression model
    predicted_price = model.predict(input_final)[0]

    # Prices below 0 don't make sense, so we floor it at 0
    predicted_price = max(predicted_price, 0)

    # Step 5: Display the result
    st.subheader("Estimated House Price")
    st.success(f"₹ {predicted_price:.2f} Lakhs")
    st.caption("Prediction generated using Linear Regression.")

else:
    st.info("Enter the house details in the sidebar and click **Predict Price**.")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit | Model: Linear Regression | Dataset: Bengaluru House Data")
