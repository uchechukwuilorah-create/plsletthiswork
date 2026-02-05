import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Title and description
st.title("🚗 Car Price Prediction Model")
st.markdown("""
This application uses a **Random Forest Regressor** to predict car prices based on various features.
The model was trained on 1,500 car records and achieved an **R² score of 0.9788**.
""")

# Create two columns for layout
col1, col2 = st.columns(2)

# Left column - Input features
with col1:
    st.subheader("📋 Enter Car Details")
    
    brand = st.selectbox(
        "Brand",
        ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes-Benz', 'Audi', 'Volkswagen', 'Hyundai', 'Kia', 'Tesla']
    )
    
    year = st.slider(
        "Year",
        min_value=2010,
        max_value=2024,
        value=2020,
        step=1
    )
    
    mileage = st.number_input(
        "Mileage (km)",
        min_value=0,
        max_value=500000,
        value=50000,
        step=1000
    )
    
    hp = st.slider(
        "Horsepower (HP)",
        min_value=100,
        max_value=500,
        value=200,
        step=10
    )
    
    fuel_type = st.selectbox(
        "Fuel Type",
        ['Gasoline', 'Diesel', 'Electric', 'Hybrid']
    )
    
    transmission = st.selectbox(
        "Transmission",
        ['Manual', 'Automatic']
    )
    
    engine_capacity = st.slider(
        "Engine Capacity (L)",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.1
    )
    
    owners = st.slider(
        "Number of Previous Owners",
        min_value=1,
        max_value=4,
        value=1,
        step=1
    )

# Right column - Prediction and results
with col2:
    st.subheader("🎯 Prediction Results")
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Year': [year],
        'Mileage': [mileage],
        'HP': [hp],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Engine_Capacity': [engine_capacity],
        'Owners': [owners]
    })
    
    # Make prediction
    if st.button("🔮 Predict Price", use_container_width=True):
        try:
            predicted_price = model.predict(input_data)[0]
            
            # Display prediction
            st.success("Prediction Successful!")
            st.metric(
                label="Predicted Car Price",
                value=f"${predicted_price:,.2f}",
                delta=None
            )
            
            # Display input summary
            st.subheader("📊 Input Summary")
            summary_data = {
                'Feature': ['Brand', 'Year', 'Mileage', 'HP', 'Fuel Type', 'Transmission', 'Engine Capacity', 'Owners'],
                'Value': [brand, year, f"{mileage:,} km", f"{hp} HP", fuel_type, transmission, f"{engine_capacity} L", owners]
            }
            st.table(summary_data)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Footer with model information
st.divider()
st.subheader("📈 Model Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("R² Score", "0.9788")

with col2:
    st.metric("MAE", "$2,283.51")

with col3:
    st.metric("Training Samples", "1,500")

st.markdown("""
---
**Model Details:**
- **Algorithm:** Random Forest Regressor with 200 estimators
- **Features:** 8 input features (Brand, Year, Mileage, HP, Fuel Type, Transmission, Engine Capacity, Owners)
- **Preprocessing:** StandardScaler for numerical features, OneHotEncoder for categorical features
- **Performance:** Excellent predictive accuracy with R² > 0.97
""")
