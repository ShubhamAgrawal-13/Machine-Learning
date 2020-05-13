#!/usr/bin/python3

''' created by shubham '''

from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten


window_size=60



def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


def print_metrics(true_labels,predicted_labels):
  # true_labels=true_labels.flatten()
  # predicted_labels=predicted_labels.flatten()
  mse=mean_squared_error(true_labels, predicted_labels)
  print('Mean Squared Error : ',mse)
  print()
  rmse = sqrt(mse)
  print('Root Mean Squared Error : ',rmse)
  print()
  mape = mean_absolute_percentage_error(true_labels, predicted_labels)
  print('Mean Absolute Percentage Error : ',mape)
  print()

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x = sequence[i:end_ix+1]
		X.append(seq_x)
	return np.array(X)


def read_data(path):
	dataset = pd.read_csv(path, sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
	print(dataset.shape)
	X_main=dataset['Global_active_power']
	X_main=X_main.to_numpy()
	print(X_main.shape)
	return X_main


def make_timeseries_dataset(X_main):
	X= split_sequence(X_main,window_size)
	X=X.astype('str')
	mask = np.all(X !='?', axis=1)
	X=X[mask]
	XX=X[:,:-1].astype(float)
	y=X[:,-1].astype(float)
	#print(X.shape)
	X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.20, random_state=31)
	return X_train, X_test, y_train, y_test, mask

def train(X_train, y_train):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=window_size))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	model.fit(X_train, y_train, epochs=3, verbose=0)
	return model

def predict(X_test):
	y_pred = model.predict(X_test, verbose=0)
	return y_pred



def predict_missing_values(mask,X_main):
	X_main=X_main.astype('str')
	missing_indices=np.where(~mask)
	missing_indices=np.array(missing_indices).flatten()
	X_main=X_main.reshape(-1)
	pred_missing=[]
	for index in missing_indices:
		x_test=X_main[index-window_size:index]
		x_test=x_test.astype(float).reshape(1,-1)
		y_pred = model.predict(x_test, verbose=0)
		pred_missing.append(y_pred[0][0])
		X_main[index]=y_pred[0][0]
	return pred_missing

if __name__ == '__main__':
	X_main=read_data(str(argv[1]))
	X_train, X_test, y_train, y_test, mask=make_timeseries_dataset(X_main)
	model=train(X_train,y_train)
	y_pred=predict_missing_values(mask,X_train)
	for i in y_pred:
		print(i)
	# print(r2_score(y_test, y_pred))
	# print_metrics(y_test, y_pred)
	# print(predict_missing_values())

