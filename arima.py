import matplotlib.pyplot as plt
import numpy as np 
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import autocorrelation_plot
import pickle
import pandas as pd

E = []
true = []
pred = []
dfs = pickle.load(open('dfs.p', "rb" ) )

# data = np.array(dfs[1][dfs[1].marketgroup == 1].price)


# model = ARIMA(data, order=(3,1,1))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())

# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())


c = ['b','r','y','orange','g']
for i in [1,2,3,4,5]:
	data = np.array(dfs[0][dfs[0].marketgroup == i].price)


	# model = ARIMA(data, order=(3,1,0))
	# model_fit = model.fit(disp=0)
	# print(model_fit.summary())

	# residuals = pd.DataFrame(model_fit.resid)
	# residuals.plot()
	# plt.show()
	# residuals.plot(kind='kde')
	# plt.show()
	# print(residuals.describe())

	print i
	X = data
	# size = int(len(X) * 0.66)
	size = 20
	train, test = X[0:size], X[size:len(X)]
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=(3,1,0))
		model_fit = model.fit(solver = 'cg', disp=0)
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		# print('predicted=%f, expected=%f' % (yhat, obs))
	plt.plot(test, color = c[i-1],label = 'true price')
	plt.plot(predictions, '--', color = c[i-1],label = 'prediction')

	error = mean_squared_error(test, predictions)
	E.append(error)
	pred.append(predictions)
	true.append(test)
	# print('Test MSE: %.3f' % error)
	# plt.plot(test, label = 'true price')
	# plt.plot(predictions, color='red', label = 'prediction')
	# plt.title('Time series prediction')
plt.show()
print sum(E)
