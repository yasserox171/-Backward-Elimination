# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:36:31 2021

@author: asus
"""
# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
  
#importing datasets  
data_set= pd.read_csv('50_CompList.csv')  
  
#Extracting Independent and dependent Variable  
x_BE= data_set.iloc[:, :-1].values  
y_BE= data_set.iloc[:, 1].values  
  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_BE, y_BE, test_size= 0.2, random_state=0)  
  
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(nm.array(x_BE_train).reshape(-1,1), y_BE_train)  
  
#Predicting the Test set result;  
y_pred= regressor.predict(x_BE_test)  
  
#Cheking the score  
print('Train Score: ', regressor.score(x_BE_train, y_BE_train))  
print('Test Score: ', regressor.score(x_BE_test, y_BE_test))  