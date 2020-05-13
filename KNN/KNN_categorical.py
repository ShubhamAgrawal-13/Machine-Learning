#!/usr/bin/python3
"""
Created on Fri Jan 31 00:42:42 2020

@author: shubham
"""

import numpy as np
import pandas as pd

original_data_labels=[ ['e','p'],
                      ['b','c','x','f','k','s'],
                      ['f','g','y','s'],
                      ['n','b','c','g','r','p','u','e','w','y'],
                      ['f','t'],
                      ['a','l','c','y','f','m','n','p','s'],
                      ['n','f','d','a'],
                      ['c','w','d'],
                      ['n','b'],
                      ['k','n','b','h','g','r','o','p','u','e','w','y'],
                      ['t','e'],
                      ['b','c','u','e','z','r'],
                      ['s','k','y','f'],
                      ['f','y','k','s'],
                      ['y','w','e','p','o','g','c','b','n'],
                      ['y','w','e','p','o','g','c','b','n'],
                      ['u','p'],
                      ['n','o','w','y'],
                      ['t','o','n'],
                      ['l','f','e','c','n','p','s','z'],
                      ['y','w','u','o','r','h','b','n','k'],
                      ['y','v','s','n','c','a'],
                      ['g','l','m','p','u','w','d'] ]


class KNNClassifier:
    K=3
    data=None
    def train(self,path):
        train_data=pd.read_csv(path,header=None)
        self.data=pd.DataFrame(columns=None)
        for i in range(23):
            df = pd.get_dummies(train_data[i], prefix='', prefix_sep='')
            df = df.T.reindex(original_data_labels[i]).T.fillna(0)
            self.data=pd.concat([self.data,df],axis=1)
            
        train, validate = np.split(pd.DataFrame(self.data).to_numpy(), [int(0.75*len(self.data))])
        #print(len(self.data),len(train),len(validate))
        maxacc=0
        for KK in range(3,10,2):
            correct=0
            wrong=0
            for i in range(len(validate)):
                dists=[]
                for j in range(len(train)):
                    dist = np.linalg.norm(validate[i][2:]-train[j][2:])
                    dists.append([dist,[train[j][0],train[j][1]]])
                dists.sort()
                edible=0
                poisonous=0
                for k in range(KK):
                    if(dists[k][1]==[1,0]):
                        edible+=1
                    else:
                        poisonous+=1
                ans=[]
                if(edible>poisonous):
                    ans=[1,0]
                else:
                    ans=[0,1]
                
                if(validate[i][0]==ans[0] and validate[i][1]==ans[1]):
                    correct+=1
                else:
                    wrong+=1

            acc=correct/1124*100
            #print(KK,acc)
            if(acc>maxacc):
            	maxacc=acc
            	self.K=KK

            
    def predict(self,path):
        test_data=pd.read_csv(path,header=None)
        test=pd.DataFrame(columns=None)
        
        for i in range(22):
            df = pd.get_dummies(test_data[i], prefix='', prefix_sep='')
            df = df.T.reindex(original_data_labels[i+1]).T.fillna(0)
            test=pd.concat([test,df],axis=1)
            
        train=pd.DataFrame(self.data).to_numpy()
        test=pd.DataFrame(test).to_numpy()
        #print(self.K)
        
        ans=[]

        for i in range(len(test)):
            dists=[]
            for j in range(len(self.data)):
                dist = np.linalg.norm(test[i]-train[j][2:])
                dists.append([dist,[train[j][0],train[j][1]]])
            dists.sort()
            edible=0
            poisonous=0
            for k in range(self.K):
                #print(dists[k][1])
                if(dists[k][1]==[1,0]):
                    edible+=1
                else:
                    poisonous+=1
            #print(edible,poisonous)
            if(edible>poisonous):
                ans.append('e')
            else:
                ans.append('p')

        return ans