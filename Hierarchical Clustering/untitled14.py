# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 07:16:34 2022

@author: Dell
"""

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
# Importing the dataset
customer_data = pd.read_csv('hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv')
customer_data.shape
customer_data.head()
data = customer_data.iloc[:, 3:5].values
#know the clusters that we want our data to be split to
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
#know the number of clusters for our dataset
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
#plot the clusters to see how actually our data has been clustered
plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')

