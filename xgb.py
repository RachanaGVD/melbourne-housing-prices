import pandas as pd
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

# Read the data
data = pd.read_csv('melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# Define a model using XGBRegressor
my_model = XGBRegressor()
my_model.fit(X_train, y_train)
print(X_train.head())