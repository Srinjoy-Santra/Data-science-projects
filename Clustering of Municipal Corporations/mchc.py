# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:08:02 2019

@author: Srinjoy Santra
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('municipalcorps.xls')

headers=['Local Body','Income','Expenditure']
df = pd.DataFrame(columns=headers)

detail = dataset.iloc[:,[0,1,2]].values.tolist()

bodydict={}
for i in range(0,len(detail),2):
    bodydict[detail[i][0]]=[detail[i][2],detail[i+1][2]]
    
names = list(bodydict)
money =[]
for name in names:
    money.append(bodydict[name])

X = np.array(money)
X = X[~np.isnan(X)]
X = X.reshape(54,2)

'''N=54
x = X[:,0]
y = X[:,1]
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.plot(x, y,'bo')
plt.show()
'''

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 20, c = 'yellow', label = 'Centroids')
plt.title('K-Means Clusters of Municipalities (2001-02)')
plt.xlabel('Annual Income (Rs. in Crores)')
plt.ylabel('Annual Expenditure (Rs. in Crores)')
plt.legend()
plt.show()

 
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Municipal corporations')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
#plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Agglomerative Clusters of Municipalities (2001-02)')
plt.xlabel('Annual Income (Rs. in Crores)')
plt.ylabel('Annual Expenditure (Rs. in Crores)')
plt.legend()
plt.show()



    
    
