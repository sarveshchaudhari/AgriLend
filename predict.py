import joblib
import numpy as np
import pandas as pd

model = joblib.load("random_forest_model.pkl")

sample_input = {
    'Land_Size': 10,  # in acres
    'Loan_Amount': 50000,  # in the same unit as used during training
    'Location': 1,  # encoded using LabelEncoder
    'Soil_Type': 2,  # encoded using LabelEncoder
    'Risk_Factor': 1,  # encoded using LabelEncoder
    'Temperature': 30,  # in Celsius
    'Rainfall': 200,  # in mmlabel 
    'Humidity': 60,  # percentage
    'Wind_Speed': 15  # in km/h
}

sample_df = pd.DataFrame([sample_input])
predicted_loan_amount = model.predict(sample_df)

print("Predicted Maximum Loan Amount:", predicted_loan_amount[0])