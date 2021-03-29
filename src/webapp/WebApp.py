# TODO: Description

import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from joblib import load
from sklearn.preprocessing import PolynomialFeatures

# Title
st.title('Barcelona Apartments Rental Price Predictor')

# Open and display an image
image = Image.open('data/images/barcelona_long.png')
st.image(image, use_column_width=True)

# Sub-title
st.write('Predict the rental prices of apartments in Barcelona entering the information on the sidebar.')

# Get user input
def get_user_input():
    rooms = st.sidebar.slider('Rooms:', 1, 11, 2)
    bathrooms = st.sidebar.slider('Bathrooms:', 1, 8, 1)
    sizem2 = st.sidebar.number_input('Size in m2:', 11, 1100, 75)
    district = st.sidebar.radio('District:', ['Eixample','Ciutat Vella','Horta-Guinardo',
                                                 'Gracia','Les Corts','Nou Barris','Sant Andreu',
                                                 'Sant Marti','Sants Montjuic','Sarria - Sant Gervasi'])

    # Store data into dict
    user_data = {'rooms': rooms, 'bathrooms': bathrooms, 'sizem2': sizem2, 'Eixample': 0, 'Ciutat Vella': 0,
                 'Horta-Guinardo': 0, 'Gracia': 0, 'Les Corts': 0, 'Nou Barris': 0, 'Sant Andreu': 0, 'Sant Marti': 0,
                 'Sants Montjuic': 0, 'Sarria - Sant Gervasi': 0, district: 1}

    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])

    # Add second degree polynomials
    poly = PolynomialFeatures(degree=2, include_bias=False)
    features_poly = poly.fit_transform(features)

    return features_poly


# Store user input into variable
user_input = get_user_input()

# Store the models predictions in a variable
model = load('data/models/ridge_model.joblib')
prediction = model.predict(user_input)

# Display prediction
st.subheader('Prediction: ')
st.write('# ' + "{:.0f}".format(prediction[0]) + " €")

# Show model metrics
st.subheader('')
st.write('Mean absolute error: ' + str(328) + '€')
st.write('Model : 2-degree Ridge Regression with a Cholesky solver and an α value of 14.')
st.write('Data from December 2020.')
st.write('Created by [aayzaa](https://github.com/aayzaa).')

