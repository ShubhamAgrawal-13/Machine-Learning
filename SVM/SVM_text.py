import numpy as np
import pandas as pd
import re
import string
from sklearn import svm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

import nltk
nltk.download('punkt')

porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def remove_punctuation(line):
    return line.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(line):
    return re.sub(r'\d+', '', line)

def to_replace_newline(line):
    return line.replace('\n',' ')
  
def remove_spaces(line):
    return " ".join(line.split(' '))

class AuthorClassifier():
	def __init__(self):
		super(AuthorClassifier, self).__init__()

	def train(self,path):
		dataset=pd.read_csv(path)
		y=dataset['author']
		listt=[]
		for i in range(dataset.shape[0]):
		    line=to_replace_newline(dataset['text'][i])   #replace newline with space
		    line=remove_punctuation(line) #remove punctuation
		    line=remove_spaces(line)    #remove extra spaces
		    line=line.replace('\s',' ')
		    line = remove_numbers(line) #remove digits
		    line=stemSentence(line)  #stemming
		    listt.append(line)
		self.word_vectorizer = TfidfVectorizer(min_df=1,stop_words='english')

		self.word_vectorizer.fit(listt)
		features = self.word_vectorizer.transform(listt)
		df_vectors = features.toarray()
		vocabulary = self.word_vectorizer.get_feature_names()
		len(vocabulary)
		dataset = pd.concat(
							    [
							        dataset,
							        pd.DataFrame(
							            df_vectors, 
							            index=dataset.index, 
							            columns=vocabulary
							        )
							    ], axis=1
							)

		dataset=dataset.drop([dataset.columns[0],dataset.columns[1],dataset.columns[2]], axis = 1) 
		dataset=dataset.values
		y=y.values

		self.pca = PCA(n_components=500)
		self.pca.fit(dataset,y)
		dataset=self.pca.transform(dataset)
		self.lin_clf = svm.LinearSVC(C=0.1)
		self.lin_clf.fit(dataset, y)

	def predict(self,path):
		X_test=pd.read_csv(path)
		#print(X_test)
		listt=[]
		for i in range(X_test.shape[0]):
		    line=to_replace_newline(X_test['text'][i])   #replace newline with space
		    line=remove_punctuation(line) #remove punctuation
		    line=remove_spaces(line)    #remove extra spaces
		    line=line.replace('\s',' ')
		    line = remove_numbers(line) #remove digits
		    line=stemSentence(line)  #stemming
		    listt.append(line)
		
		features = self.word_vectorizer.transform(listt)
		df_vectors = features.toarray()
		vocabulary = self.word_vectorizer.get_feature_names()
		X_test = pd.concat(
							    [
							        X_test,
							        pd.DataFrame(
							            df_vectors, 
							            index=X_test.index, 
							            columns=vocabulary
							        )
							    ], axis=1
						   )

		X_test=X_test.drop([X_test.columns[0],X_test.columns[1],X_test.columns[2]], axis = 1) 
		X_test=X_test.values

		X_test=self.pca.transform(X_test)

		prediction = self.lin_clf.predict(X_test)

		return prediction

	

