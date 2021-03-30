import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.spatial import distance
import time





# Plot all features with the target
def plotAllFeatures(concrete, regressand):
	x_label = ['feature 1', 'feature 2', 'feature 3', 'feature 4', 'feature 5', 'feature 6', 'feature 7', 'feature 8']

	for i in range(8):
		regressor = concrete.iloc[:, i]
		plt.scatter(regressor, regressand)
		plt.xlabel(x_label[i])
		plt.ylabel('target')
		plt.show()





# Problem One
def problemOne(concrete, regressand):
	print(" *** Problem One")

	scaler = StandardScaler()
	regressand = scaler.fit_transform(regressand)
	r2 = []
	weight = []
	bias = []
	mse = []
	for i in range(8):
		regressor = concrete.iloc[:, i].values.reshape(-1, 1)
		regressor = regressor.astype(float)
		regressor = scaler.fit_transform(regressor)
		plt.scatter(regressor, regressand)
		plt.show()

		x_train, x_test, y_train, y_test = train_test_split(regressor, regressand, test_size = 0.2)
		lr = LinearRegression()
		lr.fit(x_train, y_train)
		pred = lr.predict(x_test)

		r2.append(r2_score(y_test, pred))
		weight.append(lr.coef_)
		bias.append(lr.intercept_)
		mse.append(mean_squared_error(y_test, pred))

		plt.scatter(x_test, y_test)
		plt.plot(x_test, pred, color = 'red')
		plt.show()

	print("R2: ", r2, "\n")
	print("Weight: ", weight, "\n")
	print("Bias: ", bias, "\n")
	print("MSE: ", mse)

	print(" *** End of PONE\n\n")





# Problem Two
def problemTwo(concrete, regressand):
	print(" *** Problem Two")

	r2 = []
	weight = []
	bias = []
	mse = []
	scaler = StandardScaler()
	regressand = scaler.fit_transform(regressand)

	for i in range(8):
		regressor = concrete.iloc[:, i]
		regressor = regressor.astype(float)
		regressor = np.c_[np.ones(len(regressand)), regressor]
		regressor = scaler.fit_transform(regressor)
		for j in range(len(regressor)):
			regressor[j][0] = 1.0

		x_train, x_test, y_train, y_test = train_test_split(regressor, regressand, test_size = 0.2)

		init_theta = np.array([np.ones(2)])
		theta = gradientDescent(x_train, y_train, init_theta, 1.0e-3, 1.0e-8)
		pred = np.matmul(x_test, theta.T)
		pred_train = np.matmul(x_train, theta.T)

		r2.append(r2_score(y_test, pred))
		weight.append(theta[0][1])
		bias.append(theta[0][0])
		mse.append(mean_squared_error(y_train, pred_train))
		plotLine(x_test[:, 1], y_test, pred)

	print("R2: ", r2, "\n")
	print("Weight: ", weight, "\n")
	print("Bias: ", bias, "\n")
	print("MSE: ", mse)

	print(" *** End of PTWO\n\n")





# wait
# Problem Three
def problemThree(concrete, regressand):
	print(" *** Problem Three")

	scaler = StandardScaler()
	regressand = scaler.fit_transform(regressand)
	regressor = concrete.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]]
	regressor = regressor.astype(float)
	regressor = np.c_[np.ones(len(regressand)), regressor]
	regressor = scaler.fit_transform(regressor)
	for i in range(len(regressor)):
		regressor[i][0] = 1.0

	x_train, x_test, y_train, y_test = train_test_split(regressor, regressand, test_size = 0.2)

	init_theta = np.array([np.ones(9)])
	theta = gradientDescentSingle(x_train, y_train, init_theta, 1.0e-3, 1.0e-12)
	pred_train = np.matmul(x_train, theta.T)
	pred_test = np.matmul(x_test, theta.T)

	R2 = [r2_score(y_train, pred_train), r2_score(y_test, pred_test)]
	MSE = [mean_squared_error(y_train, pred_train), mean_squared_error(y_test, pred_test)]
	print("R2 (single): ", R2)
	print("MSE (single): ", MSE, "\n")

	init_theta = np.array([np.ones(9)])
	theta = gradientDescent(x_train, y_train, init_theta, 1.0e-3, 1.0e-12)
	pred_train = np.matmul(x_train, theta.T)
	pred_test = np.matmul(x_test, theta.T)

	R2 = [r2_score(y_train, pred_train), r2_score(y_test, pred_test)]
	MSE = [mean_squared_error(y_train, pred_train), mean_squared_error(y_test, pred_test)]
	print("R2 (all): ", R2)
	print("MSE (all): ", MSE)

	print(" *** End of PTHREE\n\n")





# Problem Four
def problemFour(concrete, regressand):
	print(" *** Problem Four")

	regressor = concrete.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]]
	regressor = regressor.astype(float)
	poly = PolynomialFeatures(3)
	regressor = poly.fit_transform(regressor)
	scaler = StandardScaler()
	regressor = scaler.fit_transform(regressor)
	for i in range(len(regressor)):
		regressor[i][0] = 1.0
	regressand = scaler.fit_transform(regressand)

	x_train, x_test, y_train, y_test = train_test_split(regressor, regressand, test_size = 0.2)

	init_theta = np.array([np.ones(165)])
	theta = gradientDescent(x_train, y_train, init_theta, 5.0e-5, 1.0e-6)
	pred_train = np.matmul(x_train, theta.T)
	pred_test = np.matmul(x_test, theta.T)

	R2 = [r2_score(y_train, pred_train), r2_score(y_test, pred_test)]
	MSE = [mean_squared_error(y_train, pred_train), mean_squared_error(y_test, pred_test)]
	print("R2: ", R2)
	print("MSE: ", MSE)

	print(" *** End of PFOUR")





# Gradient Descent w
def gradientDescent(x, y, theta, alpha, precision):
	startTime = time.localtime(time.time())
	currentTime = time.localtime(time.time())
	prev_size = 1
	while(not(currentTime.tm_min >= startTime.tm_min+2 and currentTime.tm_sec >= startTime.tm_sec) and prev_size > precision):
		prev_theta = theta
		theta = theta - alpha*np.sum((np.matmul(x, theta.T) - y)*x, axis = 0)
		prev_size = distance.euclidean(prev_theta[0], theta[0])
		currentTime = time.localtime(time.time())

	return theta





# Gradient Descent Single w
def gradientDescentSingle(x, y, theta, alpha, precision):
	startTime = time.localtime(time.time())
	currentTime = time.localtime(time.time())
	prev_size = 1
	while(not(currentTime.tm_min >= startTime.tm_min+2 and currentTime.tm_sec >= startTime.tm_sec) and prev_size > precision):
		prev_theta = theta
		for j in range(len(theta[0])):
			secondTheta = 0
			for i in range(len(x[:, 0])):
				secondTheta = secondTheta + x[i][j]*(y[i][0] - np.matmul(x[i], theta[0].T))
			theta[0][j] = theta[0][j] + alpha*secondTheta

		prev_size = distance.euclidean(prev_theta[0], theta[0])
		currentTime = time.localtime(time.time())

	return theta





# Plot the linear line
def plotLine(x, y, pred_y):
	plt.scatter(x, y)
	plt.plot(x, pred_y, color = 'red')
	plt.show()





if __name__ == '__main__':
	concrete = pd.read_csv('Concrete_Data.csv')
	regressand = concrete.iloc[:, 8].values.reshape(-1, 1)

	#plotAllFeatures(concrete, regressand)

	#problemOne(concrete, regressand)

	problemTwo(concrete, regressand)

	#problemThree(concrete, regressand)

	#problemFour(concrete, regressand)
