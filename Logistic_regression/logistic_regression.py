#!/usr/bin/python3

''' created by shubham '''


from sys import argv

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
from sklearn import metrics
import cv2
import os
from sklearn.preprocessing import StandardScaler

image_dimension=48 #(48*48)

def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48,48))
    return image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def read_data(path_train,path_test):
	file1=open(path_train,'r')
	file2=open(path_test,'r')
	data1=file1.read().split('\n')
	data2=file2.read().split('\n')

	# print(data1)
	X_train=[]
	y_train=[]
	X_test=[]
	
	for i in data1:
		if(i==''):
			continue
		temp=i.strip().split(' ')
		#print(temp)
		X_train.append(rgb2gray(read_image(temp[0])).flatten())
		y_train.append(str(temp[1]))

	for i in data2:
		if(i==''):
			continue
		#print(i.strip())
		X_test.append(rgb2gray(read_image(i.strip())).flatten())

	return np.array(X_train),np.array(y_train),np.array(X_test)

# def read_data(path='../A3/A3/dataset/'):
# 	#print("Reading the data ....")

# 	images=os.listdir(path)
# 	labels=[]
# 	dataset = []
# 	for image_name in images:
# 	    dataset.append()
# 	    temp=image_name.split('.')[0]
# 	    temp=temp.split('_')
# 	    labels.append(int(temp[0]))
# 	dataset=np.array(dataset)
# 	labels=np.array(labels)

# 	final_dataset=[]
# 	for i in range(dataset.shape[0]):
# 	    gray = rgb2gray(dataset[i])
# 	    final_dataset.append(gray.flatten())

# 	final_dataset=np.array(final_dataset)

# 	return final_dataset,labels




def split_data(X,y):
	#print("Spliting....")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=31)
	return X_train, X_test, y_train, y_test



def normalize(X_train,X_test):
	scaler = StandardScaler()
	X_train=scaler.fit_transform(X_train)

	X_test=scaler.transform(X_test)

	return X_train,X_test


def pca(X,n_c):
	#covarience matrix
    dataset_mean=np.mean(X,axis=0)
    dataset_mean.shape
    #print("PCA Transformation ... ")
    #print("Covariance matrix computation")
    covariance_matrix = (X - dataset_mean).T.dot((X - dataset_mean))/(X.shape[0]-1)
    #covariance_matrix.shape
    
    #Svd
    #print("SVD ...")
    u,s,v = np.linalg.svd(covariance_matrix)
    #print(u.shape)
    
    Y = np.matmul(X,u[:,:n_c])
    return Y,u


def pca_transform(X,u,n_c):
	Y = np.matmul(X,u[:,:n_c])
	return Y

def reconstruct_from_pca(Y,u):
    n=Y.shape[1]
    print('n : ',n)
    return np.matmul(Y,u[:,:n].T)

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(X_train,y_train,itr=5000,a=0.1):
	X_train=np.insert(X_train,0,1, axis=1)
	thetas = []
	classes = np.unique(y_train)
	costs = []
	#print('Training ...')
	m=len(y_train)
	for c in classes:
	    yb=np.where(y_train==c,1,0)
	    theta = np.zeros(X_train.shape[1])
	    cost=np.zeros(itr)
	    for i in range(itr):
	        h=sigmoid(np.matmul(X_train,theta))
	        cost[i]= 1/m*np.sum(-yb * np.log(h) - (1 - yb) * np.log(1 - h))
	        error=h-yb
	        diff_cost= 1/m*np.matmul(error,X_train)
	        theta = theta - a*diff_cost
	        
	    thetas.append(theta)
	    costs.append(cost)

	return costs,thetas,classes 

def predict(classes, thetas, X_test):
    X_test = np.insert(X_test, 0, 1, axis=1)
    preds = [np.argmax([sigmoid(xi @ theta) for theta in thetas]) for xi in X_test]
    return [classes[p] for p in preds]


def plot_cost_graph(itr,costs):
	x = [i for i in range(1,itr+1)]  
	y = costs[0]

	plt.plot(x, y) 
	plt.xlabel('No. of iterations') 
	plt.ylabel('cost')  
	plt.title('Cost Graph') 
	plt.show() 



if __name__ == '__main__':
	X_train, y_train, X_test =read_data(str(argv[1]),str(argv[2]))
	#print(X_train.shape)
	#print(X_test.shape)
	X_train, X_test = normalize(X_train,X_test)
	X_pca,u = pca(X_train,200)
	#print(X_pca.shape,u.shape)
	itr=2000
	a=0.01
	costs, thetas, classes=train(X_pca,y_train,itr,a)
	X_test_pca=pca_transform(X_test,u,200)
	#print(X_test_pca.shape)
	predicted_labels = predict(classes, thetas, X_test_pca)
	for i in range(len(predicted_labels)):
		print(predicted_labels[i])
	# print(accuracy_score(y_test,predicted_labels))
	# plot_cost_graph(itr,costs)







