# TODO: Description

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from joblib import dump

# Get data
df = pd.read_csv('C:/Users/Alex/PycharmProjects/barcelona-apartments-2/data/processed/barcelona_apartments2.csv')

# Create a onehot encoder category for the districts in order to do stratified train test split
df['district_cat'] = df['eixample'].astype(str) + df['ciutat_vella'].astype(str) \
                     + df['horta_guinardo'].astype(str) + df['gracia'].astype(str) \
                     + df['les_corts'].astype(str) + df['nou_barris'].astype(str) \
                     + df['sant_andreu'].astype(str) + df['sant_marti'].astype(str) \
                     + df['sants_montjuic'].astype(str) + df['sarria_sant_gervasi'].astype(str)

# Split data into a test and train set.
# The splits are stratified based on districts.
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['district_cat'])

# Delete the district encoder category
for set_ in (train, test):
    set_.drop('district_cat', axis=1, inplace=True)

# Separate labels from features
X_train = train.drop("price", axis=1)
y_train = train["price"].copy()
X_test = test.drop("price", axis=1)
y_test = test["price"].copy()

# Add second degree polynomials
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

# Create the model
ridge_model = Ridge(alpha=14, solver='cholesky')

# Train the model and save it
ridge_model.fit(X_train_poly, y_train)
dump(ridge_model, 'C:/Users/Alex/PycharmProjects/barcelona-apartments-2/data/models/ridge_model.joblib')

# Check prediction stats
predictions = ridge_model.predict(X_test_poly)
rmse = mean_squared_error(predictions, y_test, squared=False)
mae = mean_absolute_error(predictions, y_test)

print(rmse, mae)