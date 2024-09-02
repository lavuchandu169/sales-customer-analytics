

import pandas as pd
import numpy as np

file_path = 'Augmented_Retail_Data_Set_No_Time.xlsx'
data = pd.read_excel(file_path)

print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Example: Fill missing numerical values with the median
# data.fillna(data.median(), inplace=True)

# Example: Fill missing categorical values with the mode
# data['Category'] = data['Category'].fillna(data['Category'].mode()[0])

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data['SalesAmount'], kde=True, color='blue')
plt.title('Sales Amount Distribution')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
data.groupby('Date')['SalesAmount'].sum().plot()
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='SalesAmount', data=data, estimator=sum)
plt.title('Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.show()

from sklearn.model_selection import train_test_split

X = data[['QuantitySold', 'Discount', 'Price', 'Year', 'Quarter']]
y = data['SalesAmount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions using Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest - Mean Squared Error: {mse_rf:.2f}')
print(f'Random Forest - R^2 Score: {r2_rf:.2f}')

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
rf = RandomForestRegressor(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# Use the best model to predict and evaluate
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Tuned Random Forest - Mean Squared Error: {mse_rf:.2f}')
print(f'Tuned Random Forest - R^2 Score: {r2_rf:.2f}')

from sklearn.ensemble import GradientBoostingRegressor

# Initialize and train the Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)
gb_model.fit(X_train, y_train)

# Predictions using Gradient Boosting
y_pred_gb = gb_model.predict(X_test)

# Evaluate Gradient Boosting
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f'Gradient Boosting - Mean Squared Error: {mse_gb:.2f}')
print(f'Gradient Boosting - R^2 Score: {r2_gb:.2f}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, label='Random Forest', alpha=0.6)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Random Forest: Actual vs Predicted Sales')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, label='Gradient Boosting', alpha=0.6, color='red')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Gradient Boosting: Actual vs Predicted Sales')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, label='Random Forest', alpha=0.6)
plt.scatter(y_test, y_pred_gb, label='Gradient Boosting', alpha=0.6, color='red')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Random Forest vs Gradient Boosting: Actual vs Predicted Sales')
plt.legend()
plt.show()

print(f'Comparative Evaluation:\n')
print(f'Random Forest - MSE: {mse_rf:.2f}, R^2: {r2_rf:.2f}')
print(f'Gradient Boosting - MSE: {mse_gb:.2f}, R^2: {r2_gb:.2f}')

from google.colab import files
uploaded = files.upload()
