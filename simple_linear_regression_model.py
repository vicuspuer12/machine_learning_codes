# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:33:12 2022

@author: Emmanuel
"""
"Importing Libraries"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"Import Dataset"
dataset = pd.read_csv('simple_regression_dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"Spliting the Dataset into the Training set and the Test set"
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, random_state =0)

"Training the Simple linear Regression model on the Training set"
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

"Predicting the Test set result"
y_pred = regressor.predict(x_test)

"Visualising the Training set results"
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'green')
plt.title('Salary VS Years of Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

"Visualising the Test Results"
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary VS Years of Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
