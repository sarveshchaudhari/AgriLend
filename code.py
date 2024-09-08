import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score


# Function to load and prepare data
def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    le = LabelEncoder()
    data['Location'] = le.fit_transform(data['Location'])
    data['Soil_Type'] = le.fit_transform(data['Soil_Type'])
    data['Risk_Factor'] = le.fit_transform(data['Risk_Factor'])
    data['Loan_Approval'] = le.fit_transform(data['Loan_Approval'])
    data['Suggested_Crop'] = le.fit_transform(data['Suggested_Crop'])
    return data, le


# Function for model training and evaluation
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_type='regressor'):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if model_type == 'regressor':
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print("Mean Absolute Error:", mae)
        print("Root Mean Squared Error:", rmse)
        print("R² Score:", r2)
        return model, mae
    else:
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        return model


# Load and prepare data
data, label_encoder = load_and_prepare_data('farmer_loan_crop_data.csv')

# Features for loan amount prediction
features = ['Land_Size', 'Loan_Amount', 'Location', 'Soil_Type', 'Risk_Factor', 'Temperature', 'Rainfall', 'Humidity',
            'Wind_Speed']
X = data[features]
y = data['Max_Loan_Amount']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg, mae = train_and_evaluate_model(X_train, X_test, y_train, y_test, reg, model_type='regressor')

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Score (MAE):", -grid_search.best_score_)

# Use the best model from GridSearchCV
best_reg = grid_search.best_estimator_
y_pred = best_reg.predict(X_test)
print("Mean Absolute Error (Tuned Model):", mean_absolute_error(y_test, y_pred))

# Save the model
joblib.dump(best_reg, 'random_forest_model.pkl')

# Load and use the model
model = joblib.load('random_forest_model.pkl')

# Visualize feature importance
importances = best_reg.feature_importances_
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()

# Residuals and predictions plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount')
plt.show()
