#!/usr/bin/python3

''' created by shubham '''

# % tensorflow_version 1.x

import os
from sys import argv 
import codecs	

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def calc_report_metrics(testlabels,pred):
    cm=confusion_matrix(testlabels,pred)
    print('Confusion Matrix : \n\n',cm)
    print()
    accu = accuracy_score(testlabels,pred)
    print('Accuracy : ',accu)
    print()
    precision = precision_score(testlabels,pred,average='macro')
    print('Precision : ',precision)
    print()
    recall = recall_score(testlabels,pred,average='macro')
    print('Recall : ',recall)
    print()
    f1s = f1_score(testlabels,pred,average='macro')
    print('F1-Score : ',f1s)
    print()
    report=classification_report(testlabels,pred)
    print('Classification Report : \n',report)
    print()



def read_data(path):
	files = os.listdir(path)
	data_dict = {}

	for file in files:
	    if file[-5:]=='ubyte': 
	        f=open(path+file,'rb')
	        data = f.read()
	       	
	        type_of_data = get_int(data[:4])   # MAGIC NUMBER TO WHETHER IMAGE OR LABEL
	        length = get_int(data[4:8])        # 4-7: LENGTH 
	        
	        if (length==10000):
	                dataset_type = 'test'
	        elif (length==60000):
	                dataset_type = 'train'
	        
	        
	        if (type_of_data == 2051):
	            category = 'images'
	            num_rows = get_int(data[8:12])  # ROWS  
	            num_cols = get_int(data[12:16])  # COLUMNS
	            ds = np.frombuffer(data,dtype = np.uint8, offset = 16)  
	            #print(len(ds))
	            ds = ds.reshape(length,num_rows,num_cols)
	            #print(ds.shape)
	        elif(type_of_data == 2049):
	            category = 'labels'
	            ds = np.frombuffer(data, dtype=np.uint8, offset=8)
	            #print(len(ds))
	            ds = ds.reshape(length)
	            #print(ds.shape)
	            
	        data_dict[dataset_type+'_'+category] = ds

	X_train=data_dict['train_images']
	X_test=data_dict['test_images']
	y_train=data_dict['train_labels']
	y_test=data_dict['test_labels']

	return X_train, y_train, X_test, y_test

def reshape_data_cnn(len1,len2,X_train,X_test):
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	X_train_cnn=X_train.reshape(len1,28,28,1)
	X_test_cnn=X_test.reshape(len2,28,28,1)
	return X_train_cnn,X_test_cnn


def train(X_train_cnn,y_train):
	classes = 10
	Y_train = np_utils.to_categorical(y_train, classes)
	#print(Y_train.shape)


	model = Sequential()

	model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Convolution2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D((2, 2)))

	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
	model.fit(X_train_cnn,Y_train,epochs=5)
	return model



def predict(model,X_test_cnn):
	return model.predict_classes(X_test_cnn)


if __name__ == '__main__':
	X_train, y_train, X_test, y_test = read_data(str(argv[1]))
	X_train_cnn, X_test_cnn=reshape_data_cnn(len(X_train),len(X_test),X_train,X_test)
	model=train(X_train_cnn,y_train)
	predicted_classes = model.predict_classes(X_test_cnn)
	for i in predicted_classes:
		print(i)
	#calc_report_metrics(predicted_classes,y_test)




