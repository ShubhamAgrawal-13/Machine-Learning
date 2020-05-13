#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 2020

@author: shubham
"""


#importing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

#preprocessing
from sklearn.preprocessing import MinMaxScaler

#for division into train and test
from sklearn.model_selection import train_test_split

#for metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class Airfoil:
	theta=[]
	a=0.11
	itr=10000
	X=[]
	def __init__(self): 
		pass

	def gradientDescent(self, X, y, theta, iterations, a):
		past_costs = []
		past_thetas = [theta]

		theta=theta.T
		y=y.reshape(len(y),1)
		# print(theta.shape)
		# print(y.shape)
		# print(X.shape)
		m=X.shape[0]
		for i in range(iterations):
		    prediction = np.dot(X, theta)
		#         print(prediction[0])
		    error = prediction - y
		#         print(error[0])
		    cost = 1/(2*m) * np.dot(error.T, error)
		    past_costs.append(cost)
		    theta = theta - (a * (1/m) * np.dot(X.T, error))
		#         print(theta)
		    past_thetas.append(theta)
		    
		return past_thetas, past_costs

	def train(self,path):
		#loading the dataset of airfoil
		dataset=pd.read_csv(path,header=None)
		
		#Separating the data and labels
		X = dataset[[0,1,2,3,4]]
		y = dataset[5]

		#converting into numpy array
		X=X.values
		y=y.values

		#Normalization using minmax
		model=MinMaxScaler()
		X_minmax = model.fit_transform(X)
		self.X=X_minmax
		X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.2, random_state=0)

		#adding the one in dataset
		one_array=np.ones((X_train.shape[0],1), dtype=float)
		X_train=np.append(one_array,X_train,axis=1)
		one_array=np.ones((X_test.shape[0],1), dtype=float)
		X_test=np.append(one_array,X_test,axis=1)

		#taking initial value of theta
		theta = [[1,0,0,0,1,1]]
		theta=np.array(theta)

		#parameters
		a=0.11
		itr=10000

		self.a=a
		self.itr=itr

		#Computing the parameters
		past_thetas, past_costs = self.gradientDescent(X_train, y_train, theta, itr, a)
		theta = past_thetas[-1]
		self.theta=theta

		#Computing metrics
		# print('Mean Squared Error train:', mean_squared_error(y_train,X_train.dot(theta)))
		# print('r2 score :', r2_score(y_train,X_train.dot(theta)))
		# print('Mean Squared Error test:', mean_squared_error(y_test,X_test.dot(theta)))
		# print('r2 score :', r2_score(y_test,X_test.dot(theta)))


	def predict(self,path):
		dataset=pd.read_csv(path,header=None)
		
		#Separating the data and labels
		X = dataset[[0,1,2,3,4]]

		#converting into numpy array
		X=X.values

		#Normalization using minmax
		model=MinMaxScaler()
		X_minmax = model.fit_transform(X)
		X=X_minmax

		#adding the one in dataset
		one_array=np.ones((X.shape[0],1), dtype=float)
		X=np.append(one_array,X,axis=1)

		return X.dot(self.theta)



#path="Datasets/Question-3/airfoil.csv"