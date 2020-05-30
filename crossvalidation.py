import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Read data
data = pd.read_csv('melb_data.csv')

# Select subset of precitors 
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Pipeline to preprocess and define Random Forest model
my_pipeline = Pipeline(steps=[
    ('preprocessor',SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])

# calculate score using cross_val_score() and multiply -1 as the score is in negative.
# sklearn has a convention where in all metrics, greater the number, the better. 
scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

# Print the MAE 
print('MAE:\n', scores)

# Take the mean of this score.
print('Mean MAE(across experiments):\n', scores.mean())



