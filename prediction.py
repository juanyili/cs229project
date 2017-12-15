import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn import linear_model
import pickle

dfs = []
m = 980
mt = 700 # training set

data = pickle.load(open('fruit1.p', "rb" ) )
n = data.shape[0] #n food groups here
data = data.transpose()

print n
c1 = [0, 1, 8,12]
data1 = data[:,c1]
c2 = [3,5,7,9]
data2 = data[:,c2]
c3 = [2,4,6,10,11,13,14]
data3 = data[:,c3]
label = ['all fats/prepared','same cluster','ridge']#, 'Ridge', 'Lasso']
r2 = np.zeros((3, n))

reg = linear_model.LinearRegression()
for i in range(n):
	train = np.concatenate((data[:mt,:i],data[:mt,i+1:]), axis = 1)
	reg.fit(train, data[:mt,i])
	test = np.concatenate((data[mt:,:i],data[mt:,i+1:]), axis = 1)
	pred = reg.predict(test)
	r2[0,i] = reg.score(test,data[mt:,i])
	# if i == 2 or i ==5:
	# 	r2[1,i] = r2[0,i]
# print "finished 0"

for i in range(len(c1)):
	train = np.concatenate((data1[:mt,:i],data1[:mt,i+1:]), axis = 1)
	reg.fit(train, data1[:mt,i])
	test = np.concatenate((data1[mt:,:i],data1[mt:,i+1:]), axis = 1)
	pred = reg.predict(test)
	r2[1,c1[i]] = reg.score(test,data1[mt:,i])

	# reg.fit(data2[:mt,:], data1[:mt,i])
	# pred = reg.predict(data2[mt:,:])
	# r2[2,c1[i]] = reg.score(data2[mt:,:],data1[mt:,i])
print "finished 1"

for i in range(len(c2)):
	train = np.concatenate((data2[:mt,:i],data2[:mt,i+1:]), axis = 1)
	reg.fit(train, data2[:mt,i])
	test = np.concatenate((data2[mt:,:i],data2[mt:,i+1:]), axis = 1)
	pred = reg.predict(test)
	r2[1,c2[i]] = reg.score(test,data2[mt:,i])

	# reg.fit(data1[:mt,:], data2[:mt,i])
	# pred = reg.predict(data1[mt:,:])
	# r2[2,c2[i]] = reg.score(data1[mt:,:],data2[mt:,i])
print "finished 2"



for i in range(len(c3)):
	train = np.concatenate((data3[:mt,:i],data3[:mt,i+1:]), axis = 1)
	reg.fit(train, data3[:mt,i])
	test = np.concatenate((data3[mt:,:i],data3[mt:,i+1:]), axis = 1)
	pred = reg.predict(test)
	r2[1,c3[i]] = reg.score(test,data3[mt:,i])

ridge = linear_model.Ridge(alpha = 0.1)
for i in range(n):
	train = np.concatenate((data[:mt,:i],data[:mt,i+1:]), axis = 1)
	ridge.fit(train, data[:mt,i])
	test = np.concatenate((data[mt:,:i],data[mt:,i+1:]), axis = 1)
	pred = ridge.predict(test)
	r2[2,i] = ridge.score(test,data[mt:,i])

# lass = linear_model.Lasso(alpha = 0.5)
# for i in range(n):
# 	train = np.concatenate((data[:mt,:i],data[:mt,i+1:]), axis = 1)
# 	lass.fit(train, data[:mt,i])
# 	test = np.concatenate((data[mt:,:i],data[mt:,i+1:]), axis = 1)
# 	pred = lass.predict(test)
# 	r2[4,i] = lass.score(test,data[mt:,i])

pickle.dump(r2, open( "fvR2.p", "wb" ) )

# plt.plot(r2[0,:],'o', label = label[0])
# plt.plot(r2[1,:],'o', label = label[1])
# plt.plot(r2[2,:], 'o',label = label[2])
# # plt.plot(r2[3,:], 'o',label = label[3])
# # plt.plot(r2[4,:], 'o',label = label[4])
# plt.ylabel("R^2 score")
# plt.xlabel("indices of different foods")
# plt.ylim(-0.4,1)
# plt.title("Performances of LR methods when predicting the prices of fats/prepared")
# plt.legend(loc = 'lower right')
# plt.show()



# plt.plot(data[mt:,0], label = 'real price')
# plt.plot(pred, label = 'predicted')
# plt.legend()
# plt.title('predict the price of fresh/frozen fruit')
# plt.xlabel('time')
# plt.ylabel('price')
# plt.show()
