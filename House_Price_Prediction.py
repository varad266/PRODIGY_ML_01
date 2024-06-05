import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the training and testing data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Print shapes and summaries of the datasets
print('Training data shape:', train_df.shape)
print('Testing data shape:', test_df.shape)
print('Training data summary:')
print(train_df.describe())
print('Testing data summary:')
print(test_df.describe())

# Visualize the relationship between features and target variable
plt.figure(figsize=(10, 6))
plt.scatter(train_df['GrLivArea'], train_df['SalePrice'], color='blue')
plt.xlabel('Living Area (sqft)')
plt.ylabel('Sale Price')
plt.title('Sale Price vs Living Area')
plt.savefig('living_area_vs_sale_price.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(train_df['BedroomAbvGr'], train_df['SalePrice'], color='blue')
plt.xlabel('Number of Bedrooms Above Ground')
plt.ylabel('Sale Price')
plt.title('Sale Price vs Number of Bedrooms Above Ground')
plt.savefig('bedrooms_vs_sale_price.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(train_df['FullBath'], train_df['SalePrice'], color='blue')
plt.xlabel('Number of Bathrooms Above Ground')
plt.ylabel('Sale Price')
plt.title('Sale Price vs Number of Bathrooms Above Ground')
plt.savefig('bathrooms_vs_sale_price.png')
plt.show()

# Split the data into features and target variable
X = train_df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = train_df['SalePrice']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print('Mean Squared Error:', mse)
print('R2 Score:', r2)

# Visualize actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.legend()
plt.savefig('actual_vs_predicted_sale_price.png')
plt.show()


# Make predictions on the test data
test_pred = model.predict(test_df[['GrLivArea', 'BedroomAbvGr', 'FullBath']])

# Save the predicted sale prices to a CSV file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_pred})
submission_df.to_csv('predicted_sale_prices.csv', index=False)
