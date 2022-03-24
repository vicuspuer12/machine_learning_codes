# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 18:06:35 2022

@author: Emmanuel
"""
"Importing Libraries"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"Importing Data Sets"
dataset = pd.read_csv('dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
"Replacing the missing data"
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
"Encoding Categorical data"
"Encoding the independent variable"
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
"Encoding the Dependent Variable"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
"Splitting the Dataset into the Training set and Test set"
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
"Feature Scaling"
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print('\n', x_train, '\n')

print(x_test, '\n')

print(y_train, '\n')

print(y_test)
