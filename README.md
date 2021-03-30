# 📈 Barcelona Rental Price Predictor 📉

_A web-based predictor of rental prices of apartments in Barcelona using machine learning._

![Image of the web app](https://github.com/aayzaa/barcelona-apartments-2/blob/master/data/images/brpp.png?raw=true)

## Try it! 💻

**Available on the following website: [barcelona-rental-prices.herokuapp.com](http://barcelona-rental-prices.herokuapp.com/).**

Select the properties of an apartment by using the sidebar widgets: number of rooms and bathrooms, the size in squared meters and the district where it is located.
After that, the model will show the rental price prediction according to the apartment you inputted.

_Note: take the results with a handful of salt: this is a personal project and it doesn't take into account the state of an apartment, the views, if it's furnished, if the neighbors are noisy or other types of variables that end up defining a price._

## Process 👩🏽‍💻

I had come up with this idea in an [earlier project](https://github.com/aayzaa/barcelona_rental_prices_no_sklearn), after having looked for a new apartment in Barcelona for quite a while. I was also learning my first machine learning algorithm (Hello Linear Regression!) so I decided that it could be a great opportunity to try it!

The prior project used a basic Linear Regression model that I implemented without sklearn library and it had some minor issues with data inconsistencies and the lack of a GUI.

In this new version I have **improved the model** to a 2-degree polynomical Ridge Regression after finding out that it's the best one performing on this data. I have also created a **Streamlit powered website** that's hosted in Heroku, in order to allow people who doesn't want to download the code to try it.

## Current results 📊

The current model being used is a **2-degree polynomial Ridge Regression**, with a Cholesky solver and an alpha of 14. The model has been selected after trying it with cross validation and several hyperparameters configurations.

The current **mean absolute error is 328€**, which sounds quite large when thinking of a 1000€ apartment. I find that in practice, with apartments that I could afford (sorry infinity pool in Sarria), the error was lower than that. The reason of this result is probably that the data I gathered contains several high priced apartments (10k to 18k), where an error of 300 seems more reasonable.

The current data holds about **15.000 apartments** from Barcelona, and it was gathered on December 2020 from Enalquiler and Idealista.

## Tools used 🛠️

* [Python](https://www.python.org/) - Language used
* [Web Scraper](https://webscraper.io/) - Scrape apartments data
* [NumPy](https://numpy.org/) - Mathematic operations
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - Data cleaning and handling operations
* [Scikit-learn](https://scikit-learn.org/stable/) - Machine learning utilities
* [Streamlit](https://streamlit.io/) - Build web app
* [Pillow](https://pypi.org/project/Pillow/) - Add Barcelona image to the web app
* [Joblib](https://joblib.readthedocs.io/en/latest/) - To store models and data
* [Heroku](https://dashboard.heroku.com/) - Host web app

## Version 📌

2.0

## Authors ✒️

* **Alex Ayza** - *Keys presser* - [aayzaa](https://github.com/aayzaa)

## License 📄

Use any of the code however you want!

---
⌨️ with ❤️ by [aayzaa](https://github.com/aayzaa)
