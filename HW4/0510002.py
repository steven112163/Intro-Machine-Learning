import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import time
import datetime





def sigmoid(z):
	return 1 / (1 + np.exp(-z))





def forward_propagate(X, theta1, theta2):
	m = X.shape[0]

	#Write codes here
	a1 = np.insert(X, 0, values = np.ones(m), axis = 1)
	z2 = a1 * theta1.T
	a2 = np.insert(sigmoid(z2), 0, values = np.ones(m), axis = 1)
	z3 = a2 * theta2.T
	h = sigmoid(z3)

	return a1, z2, a2, z3, h





def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
	m = X.shape[0]
	X = np.matrix(X)
	y = np.matrix(y)

	# reshape the parameter array into parameter matrices for each layer
	theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

	# run the feed-forward pass
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

	# compute the cost
	J = 0
	for i in range(m):
		first_term = np.multiply(-y[i, :], np.log(h[i, :]))
		second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
		J += np.sum(first_term - second_term)

	J = J / m
	J += (float(learning_rate) / (2 * m) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2))))

	return theta1, theta2, a1, z2, a2, z3, h, J





def sigmoid_gradient(z):
	return np.multiply(sigmoid(z), (1 - sigmoid(z)))





def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
	m = X.shape[0]

	#Write codes here
	theta1, theta2, a1, z2, a2, z3, h, J = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)

	delta1 = np.zeros(theta1.shape)
	delta2 = np.zeros(theta2.shape)

	for t in range(m):
		z2_colt = z2[t, :]

		d3t = h[t, :] - y[t, :]

		z2_colt = np.insert(z2_colt, 0, values = np.ones(1))
		d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2_colt))

		delta1 = delta1 + (d2t[:, 1:]).T * a1[t, :]
		delta2 = delta2 + d3t.T * a2[t, :]

	delta1 = delta1 / m
	delta2 = delta2 / m

	delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
	delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

	grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

	return J, grad





if __name__ == '__main__':
	#Be careful with the file path!
	data = loadmat('data/hw4.mat')
	startTime = time.time()
	print('Start time: ', datetime.datetime.now())

	encoder = OneHotEncoder(sparse = False)
	y_onehot = encoder.fit_transform(data['y'])

	# initial setup
	input_size = 400
	hidden_size = 25
	num_labels = 10
	learning_rate = 1

	# randomly initialize a parameter array of the size of the full network's parameters
	params = (np.random.random(size = hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
	m = data['X'].shape[0]
	X = np.matrix(data['X'])
	y = np.matrix(data['y'])

	# unravel the parameter array into parameter matrices for each layer
	theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

	# minimize the objective function
	fmin = minimize(fun = backprop, x0 = params, args = (input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method = 'TNC', jac = True, options = {'maxiter': 250})

	X = np.matrix(X)
	theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
	theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
	y_pred = np.array(np.argmax(h, axis = 1) + 1)

	correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
	accuracy = (sum(map(int, correct)) / float(len(correct)))
	elapsedTime = time.time()
	hours, rem = divmod(elapsedTime - startTime, 3600)
	minutes, seconds = divmod(rem, 60)
	print('End time: ', datetime.datetime.now())
	print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
	print ('accuracy = {0}%'.format(accuracy * 100))
