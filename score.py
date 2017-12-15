import numpy as np 
import matplotlib.pyplot as plt 
import pickle

fv = pickle.load(open('fvR2.p', "rb" ) )
gd = pickle.load(open('gdR2.p', "rb" ) )
me = pickle.load(open('meR2.p', "rb" ) )
fats = pickle.load(open('fatsR2.p', "rb" ) )

data = np.concatenate((fv,gd), axis = 1)
data = np.concatenate((data,me), axis = 1)
data = np.concatenate((data,fats), axis = 1)
bins = np.linspace(-1,1,15)
plt.hist((data[0],data[1],data[2]),alpha = 0.7, label = ('LR','Same cluster','Ridge'))
# plt.hist(data[0],bins, alpha=0.5,  label= 'LR')
# plt.hist(data[1], bins, alpha=0.5,color = 'r', label = 'Same cluster')
# plt.hist(data[2],bins, alpha=0.5, color = 'g', label = 'Ridge')
plt.xlim(-1,1)
plt.legend(loc = 'upper left')
plt.xlabel('R^2 scores')
plt.ylabel('frequency')
plt.title('Histogram of performances of regression methods')
plt.show()