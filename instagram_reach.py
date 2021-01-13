# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:39:24 2021

@author: Deven Shetty
"""

import pandas as pd
data = pd.read_csv("instagram_reach.csv")
time_since_posted  = list(data["Time since posted"])
time_without_hour  = [sub.replace('hour', '') for sub in time_since_posted]
time_without_hours =  [sub.replace('hours', '') for sub in time_since_posted]
time_in_int_format = list(map(int, time_without_hours))
data.insert(7,"time_in_int_format", time_in_int_format, True)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = data.iloc[:,[4,7] ].values
y = data.iloc[:,8].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Finding the mean squared error
from sklearn.metrics import mean_squared_error
 
mse = mean_squared_error(y_pred,y_test)
