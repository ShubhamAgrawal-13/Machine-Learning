#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 01 2020

@author: shubham
"""



#install rdkit
!wget "https://gist.githubusercontent.com/philopon/a75a33919d9ae41dbed5bc6a39f5ede2/raw/5bb62e381123558f2cc3149f9a7baeb84c90ba03/rdkit_installer.py"
!python3 rdkit_installer.py

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import mean_absolute_error, mean_squared_error
def evaluation(model, X_test, y_test):
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    print('MAE score:', round(mae, 4))
    print('MSE score:', round(mse,4))

def number_of_atoms(atom_list, df):
    for i in atom_list:
        df['num_of_{}_atoms'.format(i)] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))


from sys import argv

path_train='/content/drive/My Drive/Smai_Assignment_4/q3/train.csv'
path_test='/content/drive/My Drive/Smai_Assignment_4/q3/final_test.csv'

df= pd.read_csv(path_train)
test_data= pd.read_csv(path_test)

from rdkit import Chem 

#preprocessing

df['mol'] = df['SMILES sequence'].apply(lambda x: Chem.MolFromSmiles(x)) 
test_data['mol'] = test_data['SMILES sequence'].apply(lambda x: Chem.MolFromSmiles(x)) 

df['mol'] = df['mol'].apply(lambda x: Chem.AddHs(x))
df['num_of_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
df['num_of_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
df['binding']=df['Binding Affinity']

test_data['mol'] = test_data['mol'].apply(lambda x: Chem.AddHs(x))
test_data['num_of_atoms'] = test_data['mol'].apply(lambda x: x.GetNumAtoms())
test_data['num_of_heavy_atoms'] = test_data['mol'].apply(lambda x: x.GetNumHeavyAtoms())
test_data['binding']=test_data['Binding Affinity']


number_of_atoms(['C','O', 'N', 'Cl','S'], df)
number_of_atoms(['C','O', 'N', 'Cl','S'], test_data)


from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from rdkit.Chem import Descriptors
df['tpsa'] = df['mol'].apply(lambda x: Descriptors.TPSA(x))
df['mol_w'] = df['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
df['num_valence_electrons'] = df['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
df['num_heteroatoms'] = df['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
df['num_radical_atoms'] = df['mol'].apply(lambda x: Descriptors.NumRadicalElectrons(x))

test_data['tpsa'] = test_data['mol'].apply(lambda x: Descriptors.TPSA(x))
test_data['mol_w'] = test_data['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
test_data['num_valence_electrons'] = test_data['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
test_data['num_heteroatoms'] = test_data['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
test_data['num_radical_atoms'] = test_data['mol'].apply(lambda x: Descriptors.NumRadicalElectrons(x))

y = df['binding'].values


train_df = df.drop(columns=['SMILES sequence', 'mol', 'Binding Affinity','binding'])
test_df = test_data.drop(columns=['SMILES sequence', 'mol', 'Binding Affinity','binding'])
print(train_df.columns)



mdf= pd.read_csv(path_train)
test= pd.read_csv(path_test)
mdf['smiles']=mdf['SMILES sequence']
test['smiles']=test['SMILES sequence']
target = mdf['Binding Affinity']
mdf.drop(columns=['Binding Affinity','SMILES sequence'],inplace=True)
test.drop(columns=['Binding Affinity','SMILES sequence'],inplace=True)

mdf['mol'] = mdf['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
test['mol'] = test['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

def camel_case_split(string):
    return re.findall('[A-Z][^A-Z]*', string)


#commands to install pretrained model
!pip install git+https://github.com/samoturk/mol2vec;
!ls 
!wget https://github.com/samoturk/mol2vec/blob/master/examples/models/model_300dim.pkl?raw=true


#Pre training
from gensim.models import word2vec
model = word2vec.Word2Vec.load('model_300dim.pkl?raw=true')
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec

#Constructing sentences
mdf['sentence'] = mdf.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
mdf['mol2vec'] = [DfVec(x) for x in sentences2vec(mdf['sentence'], model, unseen='UNK')]
X = np.array([x.vec for x in mdf['mol2vec']])
y = target.values

test['sentence'] = test.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
test['mol2vec'] = [DfVec(x) for x in sentences2vec(test['sentence'], model, unseen='UNK')]
X_test = np.array([x.vec for x in test['mol2vec']])

mdf = pd.DataFrame(X)
new_df = pd.concat((mdf, train_df), axis=1)
test_t = pd.DataFrame(X_test)
new_test_df = pd.concat((test_t, test_df), axis=1)

X_train, X_t, y_train, y_test = train_test_split(new_df, y, test_size=.1, random_state=1)


#Running the model

from sklearn.svm import SVR
clf = SVR(C=200,epsilon=0.75)
clf.fit(new_df, y)
evaluation(clf, X_t, y_test)

prediction1 = clf.predict(new_test_df)
print(prediction1)


#save to csv file
# dd=test
# dd=dd.drop(columns=dd.columns)
# dd['SMILES sequence']=test['smiles']
# dd['Binding Affinity']=prediction1
# dd.to_csv('/content/drive/My Drive/Smai_Assignment_4/q3/submission.csv', index=False)