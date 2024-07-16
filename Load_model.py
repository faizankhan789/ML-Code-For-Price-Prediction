import joblib
import pandas as pd

# Load the saved model
best_model = joblib.load('best_car_price_predictor_GradientBoosting.pkl')

# Example new data (replace this with your actual new data)
new_data = pd.DataFrame({
    'Make': ['Toyota'],
    'Model': ['Corolla'],
    'Fuel Type': ['Petrol'],
    'Transmission': ['Automatic'],
    'Registered in': ['Karachi'],
    'Color': ['White'],
    'Assembly': ['Local'],
    'Body Type': ['Sedan'],
    'Model Year': [2018],
    'Mileage(km)': [50000],
    'Engine Capacity(cc)': [1800]
})

# Preprocess the new data
new_data_preprocessed = best_model['preprocessor'].transform(new_data).toarray()

# Make predictions
predicted_prices = best_model['regressor'].predict(new_data_preprocessed)

# Display the predicted prices
print("Predicted Prices (PKR lacs):", predicted_prices)
