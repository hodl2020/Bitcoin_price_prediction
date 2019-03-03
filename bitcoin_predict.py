import quandl
import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm

# df = quandl.get("WIKI/FB") #uncomment for stocks

today = datetime.datetime.now()
earlier = today - datetime.timedelta(days=1460)


df = quandl.get("BITFINEX/BTCUSD", start_date=earlier, end_date=today)

df = df.rename(columns={'Mid': 'Adj. Close'})  # comment for stocks

df = df[['Adj. Close']]

forecast_out = int(7)  # predicting 7 days into future
# label column with data shifted 7 units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)


x = np.array(df.drop(['Prediction'], 1))
x = preprocessing.scale(x)  # Scaling features to normalize the data


x_forecast = x[-forecast_out:]  # set it to last 7
x = x[:-forecast_out]  # remove last 7 from x


y = np.array(df['Prediction'])
y = y[:-forecast_out]

# splitting of data
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.20)

# Training
clf = LinearRegression()
clf.fit(x_train, y_train)

# Testing
confidence = clf.score(x_test, y_test)
print("confidence: ", confidence)

# Prediction
forecast_prediction = clf.predict(x_forecast)
print(forecast_prediction)
print('\nThis is not a financial advise.')
