# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:39:25 2022

@author: Emmanuel
"""
"Importing Libraries"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"Importing Dataset"
dataset = pd.read_csv('svr_dataset.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y),1)
"Feature Scaling"
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
"Training SVR Model on the Entire Dataset"
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)
"Predicting New result"
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))
"Visualizing SVR result"
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"Visualizing SVR result for higher resolution and smooter curve"
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()