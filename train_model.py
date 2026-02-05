import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib as jl # Using joblib for saving the model
import pickle

# Load the data
df = pd.read_csv('car_price_data.csv')

# Define features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Identify categorical and numerical columns
categorical_cols = ['Brand', 'Fuel_Type', 'Transmission']
numerical_cols = ['Year', 'Mileage', 'HP', 'Engine_Capacity', 'Owners']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score: {r2:.4f}")
print(f"Mean Absolute Error: ${mae:.2f}")

# Save the model
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as car_price_model.pkl")
