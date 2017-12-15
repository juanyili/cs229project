import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
import pickle
from sklearn.cluster import KMeans

m = 980
# mt = 700 # training set

dfs = pickle.load(open('dfs.p', "rb" ) )
n = len(np.unique(dfs[8].marketgroup))
data = np.zeros((n,168))

ind = 0
for i in np.unique(dfs[n].marketgroup):
	p = np.empty(0)
	for j in range(8,14):
		p = np.concatenate((p,dfs[j].loc[(dfs[j].marketgroup == i)].price))
	data[ind,:] = p
	ind = ind+1

print data.shape

(n,m) = data.shape
# for i in range(n):
# 	data[i,:] = data[i,:]/np.mean(data[i,:])
# newdata = np.zeros((n,m-1))
# # 
# for i in range(n):
# 	for j in range(1,m):
# 		newdata[i,j-1] = data[i,j]/np.mean(data[i,:]) - data[i,j-1]/np.mean(data[i,:]) 

kmeans = KMeans(n_clusters=4, random_state=0).fit(data[:,:int(m*0.7)])
label =  np.array(kmeans.labels_)
print label
x =  np.unique(dfs[8].marketgroup)

print x[label == 3]

