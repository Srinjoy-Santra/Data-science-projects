# -*- coding: utf-8 -*-
"""
@author: Srinjoy Santra
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\nEW u\Documents\Machine Learning A-Z Template Folder\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import   stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

def naive_bayes():
    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def logistic_regression():
    # Fitting Naive Bayes to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def svm():
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(X_train)
    x_test = sc.transform(X_test)
    
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm

def kernel_svm():
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(X_train)
    x_test = sc.transform(X_test)
    
    # Fitting Kernel SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(x_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
    accuracies.mean()
    accuracies.std()

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm

def decision_tree():
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(X_train)
    x_test = sc.transform(X_test)
    
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm

def random_forest():
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(X_train)
    x_test = sc.transform(X_test)
    
    # Fitting Decision Tree Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm

def knn():
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(X_train)
    x_test = sc.transform(X_test)
    
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm
    
def evaluate(cm,technique):   
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[0][0]
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1score = 2*precision*recall/(precision+recall)
    return [technique,accuracy, precision, recall, f1score]


techniques = []
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, naive_bayes())
techniques.append(evaluate(cm,'Naive Bayes'))
cm = confusion_matrix(y_test, logistic_regression())
techniques.append(evaluate(cm,'Logistic Regression'))
cm=svm()
techniques.append(evaluate(cm,'Support Vector Machine'))
cm=kernel_svm()
techniques.append(evaluate(cm,'Kernel SVM'))
cm=decision_tree()
techniques.append(evaluate(cm,'Decision Tree'))
#cm=
#techniques.append(evaluate(cm,'Optimized Decision Tree'))
cm=random_forest()
techniques.append(evaluate(cm,'Random Forest Tree'))
cm=knn()
techniques.append(evaluate(cm,'K Nearest neighbour'))

df = pd.DataFrame(techniques,columns = ['Technique','accuracy', 'precision', 'recall', 'f1score'])



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
