# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

"""### **Question 1 : SVM (Image Classifiction)**

Importing the libraries
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""### Function for Unpickle the file"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

"""### **Now, First we will analyze the dataset of cifar-10.**

There are 5 data_batch files which contains the images and its labels and 1 test_batch file for testing the data.

Each data_batch file contains 10,000 images with their labels.

So there are total **50,000 images**.

and test_batch contains **10,000 images **for testing.

Each image is represented as vector of length **3072**.

Image is of size 32*32*3 **bold text** (Here, each pixel has 3 colors are there i.e., red, green, Blue)


Now, First will upickle the data_batch files and test_batch files.

### Unpickling the data from the files
"""

dataset1=unpickle("/content/drive/My Drive/cifar-10-python/cifar-10-batches-py/data_batch_1")
dataset2=unpickle("/content/drive/My Drive/cifar-10-python/cifar-10-batches-py/data_batch_2")
dataset3=unpickle("/content/drive/My Drive/cifar-10-python/cifar-10-batches-py/data_batch_3")
dataset4=unpickle("/content/drive/My Drive/cifar-10-python/cifar-10-batches-py/data_batch_4")
dataset5=unpickle("/content/drive/My Drive/cifar-10-python/cifar-10-batches-py/data_batch_5")
testset=unpickle("/content/drive/My Drive/cifar-10-python/cifar-10-batches-py/test_batch")

"""Reading the data from the dataset and understand the data"""

for k,v in testset.items():
  print(k,v)

"""Separating the data and the labels"""

dataset1_data=dataset1[b'data']
dataset1_labels=dataset1[b'labels']
dataset2_data=dataset1[b'data']
dataset2_labels=dataset1[b'labels']
dataset3_data=dataset1[b'data']
dataset3_labels=dataset1[b'labels']
dataset4_data=dataset1[b'data']
dataset4_labels=dataset1[b'labels']
dataset5_data=dataset1[b'data']
dataset5_labels=dataset1[b'labels']
testdata=testset[b'data']
testlabels=testset[b'labels']

"""Running the SVM on one dataset1"""

# X_train, X_test, y_train, y_test = train_test_split(dataset1_data, dataset1_labels, test_size=0.2, random_state=0)
# from sklearn import svm
# classifier=svm.SVC(kernel='linear',C=1.0)
# classifier.fit(X_train,y_train)
# prediction=classifier.predict(X_test)
# accuracy_score = accuracy_score(y_test,prediction)
# print(accuracy_score)

"""### Concatenating all the dataset into one dataset"""

dataset = np.concatenate((dataset1_data, dataset2_data, dataset3_data, dataset4_data, dataset5_data), axis=0)
dataset
print(dataset.shape)

"""### Concatenating all the labels set into one labels set"""

labels = np.concatenate((dataset1_labels, dataset2_labels, dataset3_labels, dataset4_labels, dataset5_labels), axis=0)
labels
print(labels.shape)

"""### Function to convert cifar-10 to image"""

# imageSize, channels, classes = 10
# trainingDataSize = 50000    
# testDataSize = 10000        
# cifar to 32*32*3
def convertImages(data):
    images = np.reshape(data,(-1, 3, 32, 32))
    images = np.transpose(images, (0,2,3,1))
    return images

"""## Converting cifar-10 data to images in both dataset and test data"""

dataset=convertImages(dataset )
print(dataset.shape)
testdata=convertImages(testdata)
print(testdata.shape)

"""## **Data visualization**"""

import matplotlib.pyplot as plt

data=dataset[1234]
print(labels[1234])
plt.imshow(data)

data=testdata[1334]
print(testlabels[1334])
plt.imshow(data)

"""## Special preprocessing for the image data


## Histogram of Oriented Gradients (HOG)

It is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image.
"""

import cv2
gammaCorrection = True
hog = cv2.HOGDescriptor((32,32),(12, 12),(4,4),(2,2),18,1, -1,0,0.2,gammaCorrection,64,True)

def to_HOG(images):
    hogDescriptors = []
    for image in images:
        hogDescriptors.append( hog.compute(image) )

    hogDescriptors = np.squeeze(hogDescriptors)
    return hogDescriptors

"""## Dimensionality Reduction using PCA"""

from sklearn.decomposition import PCA
trainHogDescriptors = to_HOG(dataset)
pca = PCA(3000)
pca.fit(trainHogDescriptors)
train=pca.transform(trainHogDescriptors)

"""### Splitting the data into training and testing (validation)"""

X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=0)

"""## C=0.1"""

from sklearn import svm
# classifier=svm.SVC(kernel='linear',C=1.0)
classifier=svm.LinearSVC(C=0.1)
classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)
accu = accuracy_score(y_test,pred)
print('Accuracy : ',accu)
cm=confusion_matrix(y_test,pred)
print('Confusion Matrix : ',cm)
precision = precision_score(y_test,pred,average=None)
print('Precision : ',precision)
recall = recall_score(y_test,pred,average=None)
print('Recall : ',recall)
f1s = f1_score(y_test,pred,average=None)
print('F1-Score : ',f1s)
report=classification_report(y_test,pred)
print('classification_report',report)

"""## C=10"""

from sklearn import svm
# classifier=svm.SVC(kernel='linear',C=10)
classifier=svm.LinearSVC(C=10)
classifier.fit(train,labels)
classifier=svm.LinearSVC(C=0.1)
classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)
accu = accuracy_score(y_test,pred)
print('Accuracy : ',accu)
cm=confusion_matrix(y_test,pred)
print('Confusion Matrix : ',cm)
precision = precision_score(y_test,pred,average=None)
print('Precision : ',precision)
recall = recall_score(y_test,pred,average=None)
print('Recall : ',recall)
f1s = f1_score(y_test,pred,average=None)
print('F1-Score : ',f1s)
report=classification_report(y_test,pred)
print('classification_report',report)



"""# **Applying SVM for training the data with C=1.0**"""

from sklearn import svm
# classifier=svm.SVC(kernel='linear',C=1.0)
classifier=svm.LinearSVC(C=1)
classifier.fit(train,labels)
test = to_HOG(testdata)
test = pca.transform(test)
pred=classifier.predict(test)
accuracy_score(testlabels,pred)

"""## Resultant Metrics"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

accu = accuracy_score(testlabels,pred)
print('Accuracy : ',accu)
cm=confusion_matrix(testlabels,pred)
print('Confusion Matrix : ',cm)
precision = precision_score(testlabels,pred,average=None)
print('Precision : ',precision)
recall = recall_score(testlabels,pred,average=None)
print('Recall : ',recall)
f1s = f1_score(testlabels,pred,average=None)
print('F1-Score : ',f1s)
report=classification_report(testlabels,pred)
print('classification_report',report)

