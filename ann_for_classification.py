# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:07:40 2019

@author: USER
"""

# HEART dISEASE DIAGONOSIS

# Using ANN classifier

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Preprocessing    
dataset = pd.read_table('processed.cleveland.data', sep = ',', header = None)

X = dataset.iloc[:,:-1]


y_class = dataset.iloc[:,-1]

y = [item>0 for item in y_class]

# Replacing missing values with most frequent one
X[11].value_counts()
X[11] = X[11].map({'?':0, '1.0' : 1.0, '2.0': 2.0, '3.0' : 3.0, '0.0' : 0.0})
X[12] = X[12].map({'6.0':6.0, '3.0': 3.0, '7.0' : 7.0, '?':3.0})

X = X.values

 # Handling categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

oneHotEncoder = OneHotEncoder(categorical_features = [2])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:,1:]
 
oneHotEncoder = OneHotEncoder(categorical_features = [8])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:,1:]

oneHotEncoder = OneHotEncoder(categorical_features = [13])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:,1:]

oneHotEncoder = OneHotEncoder(categorical_features = [16])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:,1:]


# Train_test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling for categorical variable
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)





# Building an ANN mdoel for classification

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Adding input and first hidden layer
classifier.add(Dense(units = 10, activation = 'relu', kernel_initializer = 'uniform', input_dim = 18))

# second Hidden layer
classifier.add(Dense(units = 10, activation = 'relu', kernel_initializer = 'uniform'))

classifier.add(Dense(units = 10, activation = 'relu', kernel_initializer = 'uniform'))

# output Layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


import time
start = time.time()

classifier.fit(X_train, y_train, batch_size = 5, epochs = 70)

elapsed = time.time()-start
print (elapsed)


# Make predictions
y_pred = classifier.predict(X_test)

y_pred = y_pred > 0.5

from sklearn import metrics
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print (accuracies.mean())
print (metrics.confusion_matrix(y_test, y_pred))
print (metrics.accuracy_score(y_test, y_pred))

print (metrics.f1_score(y_test, y_pred))



