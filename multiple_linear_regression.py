# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:01:08 2022

@author: Emmanuel
"""
"Importing the Necessary libraries"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"Importing the Dataset"
dataset = pd.read_csv('mlr_dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
"Encoding categorical data"
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
"Splitting the dataset into training set and test set"
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"Trainng the Multiple Linear Regression model on the Training Set"
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
"Predicting the test set result"
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
