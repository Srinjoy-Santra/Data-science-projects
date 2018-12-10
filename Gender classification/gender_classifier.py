# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:00:27 2018

@author: Srinjoy Santra
Based on Decision tree Classification
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('gender.csv')
X = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 0].values
name = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = X[1:12],X[11::],y[1:12],y[11::]

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
clf = clf.fit(X_train,y_train)
dtc=clf.predict(X_test)
    
# Fitting Naive Bayes Classification to the Training set
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(X_train,y_train)
nbc=clf.predict(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf = clf.fit(X_train,y_train)
lrc=clf.predict(X_test)

# Fitting SVM Regression to the Training set
from sklearn.svm import SVC
clf = SVC(kernel='poly',random_state=0)
clf = clf.fit(X_train,y_train)
svc=clf.predict(X_test)
    
for i in range(0,5,1):
    print(name[i+11],'lrc=',lrc[i],'dtc=',dtc[i],'nbc=',nbc[i],'svc=',svc[i],y[i+11])
# Comparing their results
from sklearn.metrics import accuracy_score
print("dtc=",accuracy_score(dtc,y_test))
print("nbc=",accuracy_score(nbc,y_test))
print("lrc=",accuracy_score(lrc,y_test))
print("svc=",accuracy_score(svc,y_test))