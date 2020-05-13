#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 01 2020

@author: shubham
"""


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout,Activation
from keras.layers import Flatten,Conv2D,MaxPooling2D

import cv2
from PIL import Image
def read_image(image_path):
    image=Image.open(image_path)
    image=image.resize([640,360])
    image=np.array(image)
    return image


import os
from sys import argv

path_train='/content/drive/My Drive/Smai_Assignment_4/q2/train.csv'
path_test='/content/drive/My Drive/Smai_Assignment_4/q2/final_test.csv'
path_train_images='/content/drive/My Drive/Smai_Assignment_4/q2/ntrain/'
path_test_images='/content/drive/My Drive/Smai_Assignment_4/q2/test/'



# load train data
train=pd.read_csv(path_train)
X_names=path_train_images+train['image_file']+'.jpg'
y=train['emotion']

XX=[]
for i in range(X_names.shape[0]):
    print(i)
    image=read_image(X_names[i])
    XX.append(image)

XX=np.array(XX)
XX=XX/255
print(XX.shape)



#load test data
test=pd.read_csv(path_test)
X_names=path_test_images+test['image_file']+'.jpg'

Xt=[]
for i in range(X_names.shape[0]):
    print(i)
    image=read_image(X_names[i])
    Xt.append(image)

Xt=np.array(Xt)
Xt=Xt/255
print(Xt.shape)


#train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.20, random_state=99)



classes = 5
Y = np_utils.to_categorical(y, classes)
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)
print(Y_train.shape,Y_test.shape)


#CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', input_shape=(360, 640, 3)))
model.add(MaxPooling2D((2, 2), name='maxpool_1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
model.add(MaxPooling2D((2, 2), name='maxpool_2'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
model.add(MaxPooling2D((2, 2), name='maxpool_3'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
model.add(MaxPooling2D((2, 2), name='maxpool_4'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', name='dense_1'))
model.add(Dense(128, activation='relu', name='dense_2'))
model.add(Dense(5,activation='softmax', name='output'))
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


model.fit(XX,Y,epochs=30,validation_data=(X_test,Y_test),batch_size=100)


#prediction
predicted_classes = model.predict_classes(X_test)

from sklearn.metrics import accuracy_score
print("validation accuracy : ",accuracy_score(y_test,predicted_classes))


predicted_classes = model.predict_classes(Xt)
print(predicted_classes)