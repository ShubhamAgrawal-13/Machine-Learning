#!/usr/bin/python3
"""
Created on Fri Jan 31 00:42:42 2020

@author: shubham
"""

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def is_categorical(array):
    return array.dtype.name == 'object'

def calculate_mse(train_data):
    mincol=0
    minsplit=0
    miny=9999999999999999999
    for col in train_data.columns[:-1]:
        unique_data=train_data[col].unique()
        list_mse=[]
        for val in unique_data:
            df1=[]
            df2=[]
            if(is_categorical(train_data[col])):
                df1=train_data[train_data[col]==val]
                df2=train_data[train_data[col]!=val]
            else:
                df1=train_data[train_data[col]<=val]
                df2=train_data[train_data[col]>val]
            
            len1=df1.shape[0]
            len2=df2.shape[0]
            
            if(len1==0 or len2==0):
                list_mse.append([9999999999999999999999,val])
                continue
            
            sp1=df1['SalePrice'].to_numpy()
            sp2=df2['SalePrice'].to_numpy()
            
            mean1=np.sum(sp1)/len1
            mean2=np.sum(sp2)/len2

            mse1=np.sum((sp1-mean1)**2)
            mse2=np.sum((sp2-mean2)**2)
            
            weight_mean=(mse1*len1+mse2*len2)/len(train_data)
            list_mse.append([weight_mean,val])
        
        min_val=0
        mini=999999999999999999999
        for i in range(len(list_mse)):
            if(mini>list_mse[i][0]):
                mini=list_mse[i][0]
                min_val=list_mse[i][1]
        
        if(miny>mini):
            miny=mini
            minsplit=min_val
            mincol=col
            
    return [miny,minsplit,mincol]

class Node:
    def __init__(self,data,split,split_col,depth):
        self.left=None
        self.right=None
        self.data=data
        self.mean=0
        self.split=split
        self.split_col=split_col
        self.depth=depth
        
class DecisionTree:
    def __init__(self):
        self.root=None
        self.train_data=None
        self.test_data=None
        self.nan_col=None
    
    def getRoot(self):
        return self.root
    
    def build(self,data,depth):
        if(depth>13):
            return None
        if(data.shape[0]<4):
            return None
        m=calculate_mse(data)
        #print(m)
        col=m[2]
        df1=[]
        df2=[]
        if(is_categorical(data[col])):
            df1=data[data[col]==m[1]]
            df2=data[data[col]!=m[1]]
        else:
            df1=data[data[col]<=m[1]]
            df2=data[data[col]>m[1]]
        
        node=Node(m,m[1],m[2],depth)
        
        if(is_categorical(data[col])):
            node.mean=data['SalePrice'].mode()[0]
        else:
            node.mean=data['SalePrice'].mean()
        
        node.left=self.build(df1,depth+1)
        node.right=self.build(df2,depth+1)
        
        return node
    
    
    def predicted(self,test_data,i,root):
        if(root.left==None and root.right==None):
            return root.mean
        #print(root.split_col)
        
        val=test_data[root.split_col][i]
        
        if(is_categorical(test_data[root.split_col])):
            if(val==root.split):
                if(root.left==None):
                    return root.mean
                else:
                    return self.predicted(test_data,i,root.left)
            else:
                if(root.right==None):
                    return root.mean
                else:
                    return self.predicted(test_data,i,root.right)
        else:
            if(val<=root.split):
                if(root.left==None):
                    return root.mean
                else:
                    return self.predicted(test_data,i,root.left)
            else:
                if(root.right==None):
                    return root.mean
                else:
                    return self.predicted(test_data,i,root.right)
        
    def predict(self,path):
        test_data=pd.read_csv(path)
        nan_col=self.nan_col
        test_data.drop(nan_col, axis = 1, inplace = True)
        for col in test_data:
            if(is_categorical(test_data[col])):
                test_data[col].fillna(test_data[col].mode()[0],inplace=True)
            else:
                test_data[col].fillna(test_data[col].mean(),inplace=True)
                
        result=[]
        for i in range(test_data.shape[0]):
            #print("--------")
            result.append(self.predicted(test_data,i,self.root))
        return result
    
    
    def train(self,path):
        self.train_data=pd.read_csv(path)
        nan_col=['Id']
        threshold=int(self.train_data.shape[0]*0.55)
        
        for col in self.train_data:
            no_of_nan=len(self.train_data[col])-self.train_data[col].count()
            if(no_of_nan>threshold):
                nan_col.append(col)
        
        self.train_data.drop(nan_col, axis = 1, inplace = True)
        self.nan_col=nan_col
        
        for col in self.train_data.columns[:-1]:
            if(is_categorical(self.train_data[col])):
                self.train_data[col].fillna(self.train_data[col].mode()[0],inplace=True)
            else:
                self.train_data[col].fillna(self.train_data[col].mean(),inplace=True)
        
        self.root=self.build(self.train_data,0)
    
if __name__=="__main__":
    dt=DecisionTree()
    dt.train('./Datasets/q3/train.csv')
    predictions=dt.predict('./Datasets/q3/test.csv')
    test_labels = list()
    with open("./Datasets/q3/test_labels.csv") as f:
      for line in f:
        test_labels.append(float(line.split(',')[1]))
    print(mean_squared_error(test_labels, predictions))
    print(r2_score(test_labels,predictions))
    

