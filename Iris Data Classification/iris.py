# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:29:19 2018

@author: nEW u

Problem: Predict the class of the flower based on available attributes.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

headers = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('iris.csv',names=headers)

print(dataset.shape)

# Univariate plot to depict each attribute
dataset.hist()
plt.show()

# Multivariate plot to depict relations among attributes
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 7)

# Building models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=0)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

clf = KNeighborsClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))

for i in range(0,len(X_test)):
    print(X_test[i],y_test[i])