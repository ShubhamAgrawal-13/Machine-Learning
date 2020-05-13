#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 01 2020

@author: shubham
"""


#importing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

path_train='/content/drive/My Drive/Smai_Assignment_4/q1/train.csv'
path_test='/content/drive/My Drive/Smai_Assignment_4/q1/final_test.csv'

dataset_train=pd.read_csv(path_train)
dataset_test=pd.read_csv(path_test)
print(dataset_train.shape)
print(dataset_test.shape)

X=dataset_train['text']
X_test=dataset_test['text']
y=dataset_train['labels']


#Preprocessing

import string
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
  
def remove_punctuation(line):
    return line.translate(str.maketrans('', '', string.punctuation))

def remove_spaces(line):
    return " ".join(line.split(' '))

#for stemming the data
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import nltk
nltk.download('wordnet')

porter=PorterStemmer()

def remove_stopwords(sentence):
  token_words=word_tokenize(sentence)
  stem_sentence=[]
  for word in token_words:
      if not word in stopwords.words():
        stem_sentence.append(word)
  return " ".join(stem_sentence)

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    # token_words
    stem_sentence=[]
    for word in token_words:
      stem_sentence.append(porter.stem(word))
    return " ".join(stem_sentence)

def pre_process(X):
  list_X=[]
  for i in range(X.shape[0]):
    comment=X[i]
    comment=comment.lower()
    comment = re.sub(r"http\S+", "", comment)
    comment=remove_punctuation(comment)
    comment=''.join([i for i in comment if not i.isdigit()])
    comment=remove_stopwords(comment)
    comment=stemSentence(comment)
    list_X.append(comment)

  print(len(list_X))
  return list_X

list_X=pre_process(X)
XX=np.array(list_X)

list_X_test=pre_process(X_test)
X_t=np.array(list_X_test)


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
word_vectorizer = TfidfVectorizer()
word_vectorizer.fit(XX)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.20, random_state=99)

X_train_tfidf = word_vectorizer.transform(XX).toarray()
X_test_tfidf = word_vectorizer.transform(X_test).toarray()

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train_tfidf, y)
predicted=clf.predict(X_test_tfidf)

from sklearn.metrics import accuracy_score
print("Validation Accuracy : ",accuracy_score(y_test, predicted))

X_test_tfidf = word_vectorizer.transform(X_t).toarray()
predicted=clf.predict(X_test_tfidf)
print(predicted)