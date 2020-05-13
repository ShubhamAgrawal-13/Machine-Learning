#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 29 2020

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

#for vectorizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#for formatted date time
from datetime import datetime
import time

class Weather:
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
		dataset=pd.read_csv(path)
		
		#labels
		y=dataset['Apparent Temperature (C)']

		#dataset['Summary'].unique()
		# dataset.isnull().sum()
		# dataset.columns
		dataset["Precip Type"].fillna("nothing", inplace = True) 
		# dataset.isnull().sum()
		dataset=pd.concat([dataset,pd.get_dummies(dataset['Precip Type'])],axis=1)
		dataset=dataset.drop(['Precip Type'], axis = 1) 
		dataset=dataset.drop(['Apparent Temperature (C)'], axis = 1) 

		# for i in range(dataset.shape[0]):
		#     s = str(dataset['Formatted Date'][i])
		#     d = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f %z")
		#     # print(time.mktime(d.timetuple()))
		# #     print(d.month)
		#     dataset['Formatted Date'][i]=d.month
		# # dataset
		# dataset=pd.concat([dataset,pd.get_dummies(dataset['Formatted Date'])],axis=1)
		dataset=dataset.drop(['Formatted Date'], axis = 1)
		dataset=dataset.drop(['Summary','Daily Summary'], axis = 1) 
		#Normalization using minmax
		model=MinMaxScaler()
		X_minmax = model.fit_transform(dataset)
		self.X=X_minmax
		y=y.values

		X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.2, random_state=0)
		#X_minmax.shape

		
		#adding the one in dataset
		one_array=np.ones((X_train.shape[0],1), dtype=float)
		X_train=np.append(one_array,X_train,axis=1)
		one_array=np.ones((X_test.shape[0],1), dtype=float)
		X_test=np.append(one_array,X_test,axis=1)

		#taking initial value of theta
		theta = [[1 for i in range(X_train.shape[1])]]
		theta=np.array(theta)

		#parameters
		a=0.1
		itr=5000

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
		X=pd.read_csv(path)
		X["Precip Type"].fillna("nothing", inplace = True) 
		X=pd.concat([X,pd.get_dummies(X['Precip Type'])],axis=1)
		X=X.drop(['Precip Type'], axis = 1)
		X=X.drop(['Formatted Date'], axis = 1)
		X=X.drop(['Summary','Daily Summary'], axis = 1) 
		X=X.drop(['Apparent Temperature (C)'], axis = 1) 

		#Normalization using minmax
		model=MinMaxScaler()
		X_minmax = model.fit_transform(X)
		X=X_minmax

		one_array=np.ones((X.shape[0],1), dtype=float)
		X=np.append(one_array,X,axis=1)
		return np.dot(X,self.theta)

path="Datasets/Question-4/weather.csv"