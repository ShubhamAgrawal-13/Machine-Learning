import os
import re
import string

#common libraries
import numpy as np
import pandas as pd

#To compute euclidean distance
from scipy.spatial import distance

#for calculating the metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import homogeneity_score

#for feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def to_decode(df,i):
    return df[0][i].decode("utf-8",errors="ignore")

def to_replace_newline(line):
    return line.replace('\n',' ')

def remove_punctuation(line):
    return line.translate(str.maketrans('', '', string.punctuation))

def remove_spaces(line):
    return " ".join(line.split(' '))

def remove_numbers(line):
    return re.sub(r'\d+', '', line)



class Cluster(object):
	"""docstring for Cluster"""
	def __init__(self):
		pass

	def cluster(self,path):
		filenames = os.listdir(path)
		original_clusters=[[],[],[],[],[],[]]
		actual_labels=[]
		list_files=[]
		for filename in filenames:
		    temp1=filename.split('.')[0].split('_')
		    #print(temp1)
		    index=int(temp1[1])
		    num=int(temp1[0])
		    original_clusters[index].append(num)
		    actual_labels.append(index)
		    list_files.append(filename)

		# for i in range(1,6):
		# 	print('cluster ',i,'-',len(original_clusters[i]))

		data = []
		files = os.listdir(path)

		for f in files:
		    fn=path+"/"+f
		    temp1=f.split('.')[0].split('_')
		    #print(temp1)
		    index=int(temp1[1])
		    num=int(temp1[0])
		    with open (fn, "rb") as myfile:
		        data.append(myfile.read())
		        
		df = pd.DataFrame(data)
		listt=[]
		for i in range(df.shape[0]):
		    line=to_decode(df,i) #decoding from bytes to string
		    line=to_replace_newline(line)   #replace newline with space
		    line=remove_punctuation(line) #remove punctuation
		    line=remove_spaces(line)    #remove extra spaces
		    line=line.replace('\s',' ')
		    line = remove_numbers(line) #remove digits
		    line=stemSentence(line)  #stemming
		    listt.append(line)

		word_vectorizer = TfidfVectorizer(stop_words='english',lowercase='True')
		word_vectorizer.fit(listt)
		df_X_train_word_features = word_vectorizer.transform(listt)
		df_vectors = df_X_train_word_features.toarray()
		vocabulary = word_vectorizer.get_feature_names()
		#print(vocabulary)
		# print(len(vocabulary))

		n=5
		index = np.random.choice(df_vectors.shape[0], n, replace=False) 
		# print(index)
		#index=[118,784,1338,1516,1151]
		# centers=df_vectors[index]
		# print(centers)
		centers=df_vectors[index]
		# centers=np.random.rand(5,len(df_vectors[0]))*np.std(df_vectors)
		# print(centers)
		clusters=[]
		predicted_labels=[]

		for itr in range(30):
		    clusters=[[],[],[],[],[]]
		    predicted_labels=[]
		    for i in range(df_vectors.shape[0]):
		        dist=[]
		        for j in range(5):
		            d = distance.euclidean(df_vectors[i], centers[j])
		            dist.append(d)
		        
		        predicted_labels.append((np.argmin(dist,axis=0)))
		        clusters[(np.argmin(dist,axis=0))].append(df_vectors[i])
		        
		    
		    for i in range(5):
		         # print(clusters[i])
		         centers[i]=np.mean(clusters[i],axis=0)
		    # print(itr," : ")
		    # for i in range(5):
		    #     print(len(clusters[i]),end=' ')
		    # print()

		mapping={}
		for i in range(len(predicted_labels)):
		    mapping[list_files[i]]=predicted_labels[i]

		print(homogeneity_score(actual_labels,predicted_labels))
		    
		return mapping


