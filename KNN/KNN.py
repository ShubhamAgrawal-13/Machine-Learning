#!/usr/bin/python3
"""
Created on Fri Jan 31 00:42:42 2020

@author: shubham
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class KNNClassifier:
    K=3
    dataset_train=None
    def train(self,path):
        self.dataset_train = pd.read_csv(path,header=None)
        self.dataset_train=pd.DataFrame(self.dataset_train).to_numpy()
        
        train, validate = np.split(self.dataset_train, [int(0.8*20000)])
        maxacc=0
        for KK in range(3,3,2):
            correct=0
            wrong=0
            for entry in validate:
                #print(entry)
                dists=[]
                for train_entry in train:
                    dist = np.linalg.norm(entry[1:]-train_entry[1:])
                    dists.append([dist,train_entry[0]])
                dists.sort()
                
                map={}
                for k in range(KK):
                    if(dists[k][1] in map):
                        map[dists[k][1]]+=1
                    else:
                        map[dists[k][1]]=0
                maxvalue=-1
                maxk=0
                for k,v in map.items():
                    if(v>maxvalue):
                        maxvalue=v
                        maxk=k
                    
                if(entry[0]==maxk):
                    correct+=1
                else:
                    wrong+=1
                        
            #print(correct,wrong)
            acc=correct/4000*100
            print(KK,acc)
            if(acc>maxacc):
            	maxacc=acc
            	self.K=KK
        
        
        
    
    def predict(self,path):
        #self.dataset_train = pd.read_csv('/content/drive/My Drive/q1/train.csv',header=None)
        #self.dataset_train=pd.DataFrame(self.dataset_train).to_numpy()
        test=pd.DataFrame(pd.read_csv(path,header=None)).to_numpy()
        ans=[]
        for entry in test:
            #print(entry)
            dists=[]
            for train_entry in self.dataset_train:
                dist = np.linalg.norm(entry[:]-train_entry[1:])
                dists.append([dist,train_entry[0]])
            dists.sort()
            
            map={}
            for k in range(self.K):
                if(dists[k][1] in map):
                    map[dists[k][1]]+=1
                else:
                    map[dists[k][1]]=0
            maxvalue=-1
            maxk=0
            for k,v in map.items():
                if(v>maxvalue):
                    maxvalue=v
                    maxk=k
                
            ans.append(maxk)
        return ans
                    

if __name__== "__main__":
    knn=KNNClassifier()
    knn.train('/content/drive/My Drive/q1/train.csv')
    predictions=knn.predict('/content/drive/My Drive/q1/test.csv')
    test_labels = list()
    with open("/content/drive/My Drive/q1/test_labels.csv") as f:
      for line in f:
        test_labels.append(line.strip())
    print (accuracy_score(test_labels, predictions))