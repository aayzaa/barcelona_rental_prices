# TODO: Description

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from PIL import Image
import streamlit as st

# Title and sub-title
st.write("""
# Barcelona Apartments Price Predictor
Predict the prices of apartments in Barcelona!
""")

# Open and display an image
image = Image.open('./barcelona.jpg')
st.image(image, caption="Barcelona", use_column_width=True)

# Get data
df = pd.read_csv('./barcelona_apartments2.csv')

# Set a subheader
st.subheader('Data Information:')

# Show data as a table
st.dataframe(df.head(20))

# Show statistics
st.write(df.describe())

# Show data as a chart
chart = st.bar_chart(df.head(20))

# Split data into X and Y
X = df.iloc[:, 1:]
y = df.iloc[:, 0:1]

# Split data into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get user input
def get_user_input():
    rooms = st.sidebar.slider('rooms', 0, 10, 2)
    bathrooms = st.sidebar.slider('bathrooms', 0, 10, 1)
    sizem2 = st.sidebar.slider('sizem2', 20, 200, 70)
    district = st.sidebar.selectbox('district', ['eixample','ciutat_vella','horta_guinardo',
                                                 'gracia','les_corts','nou_barris','sant_andreu',
                                                 'sant_marti','sants_montjuic','sarria_sant_gervasi'])

    # Store data into dict
    user_data = {'rooms': rooms, 'bathrooms': bathrooms, 'sizem2': sizem2, 'eixample': 0, 'ciutat_vella': 0,
                 'horta_guinardo': 0, 'gracia': 0, 'les_corts': 0, 'nou_barris': 0, 'sant_andreu': 0, 'sant_marti': 0,
                 'sants_montjuic': 0, 'sarria_sant_gervasi': 0, district: 1}

    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store user input into variable
user_input = get_user_input()

# Set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

# Train
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Show model metrics
st.subheader('Model Error:')
st.write(str(mean_absolute_error(y_test, lr_model.predict(X_test))) + '€')

# Store the models predictions in a variable
prediction = lr_model.predict(user_input)

# Display prediction
st.subheader('Prediction: ')
st.write('# ' + str(prediction[0,0]))
