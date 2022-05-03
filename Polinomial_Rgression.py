# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:07:42 2022

@author: Emmanuel
"""
"Importing the Libraries"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"Imorting Dataset"
dataset = pd.read_csv('pr_dataset.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
"Train Linear Regression Model for the entire Dataset"
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
"Train Polinomial Regression Model for the Entire Dataset"
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
"The chart Result of the Linear Regression"
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or False (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
"The Chart Result of the Polinomial Regression"
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"Polinomial Regression chart result for Higher resolution and smoother curve"
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"Predicting a new Linear regression Result"
lin_reg.predict([[6.5]])
"Predicting New Polinomial Regression Result"
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))