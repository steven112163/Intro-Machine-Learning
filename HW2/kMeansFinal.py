from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter





# Get most common element
def mostCommon(lst):
	data = Counter(lst)
	return max(lst, key = data.get)





# Euclidean Distance Calculator
def getDistance(a, b, ax = 1):
    return np.linalg.norm(a - b, axis = ax)





# Get final centroid
def getFinalCentroid(centroid, pair, clusters, k):
	old_centroid = np.zeros(centroid.shape)
	# distance between old centroid and current centroid
	cen_distance = getDistance(centroid, old_centroid, None)

	# Loop will run till the error becomes zero
	while cen_distance != 0:
		# Assigning each value to its closest cluster
		for i in range(len(pair)):
			distances = getDistance(pair[i], centroid)
			cluster = np.argmin(distances)
			clusters[i] = cluster
		# Storing the old centroid values
		old_centroid = deepcopy(centroid)
		# Finding the new centroids by taking the average value
		for i in range(k):
			points = [pair[j] for j in range(len(pair)) if clusters[j] == i]
			if len(points) > 0:
				centroid[i] = np.mean(points, axis = 0)
		cen_distance = getDistance(centroid, old_centroid, None)

	return centroid, clusters




# Compute accuracy
def computeAccuracy(clusters, pitch_type):
	CH = [int(clusters[i]) for i in range(len(pitch_type)) if pitch_type[i] == 'CH']
	CU = [int(clusters[i]) for i in range(len(pitch_type)) if pitch_type[i] == 'CU']
	FF = [int(clusters[i]) for i in range(len(pitch_type)) if pitch_type[i] == 'FF']

	count = CH.count(mostCommon(list(CH)))
	count += CU.count(mostCommon(list(CU)))
	count += FF.count(mostCommon(list(FF)))

	return count / len(pitch_type)





# Print the scatter
def printScatter(centroid, pair, clusters, k):
	colors = ['red', 'green', 'blue', 'yellow', 'purple']
	fig, ax = plt.subplots()
	for i in range(k):
		points = np.array([pair[j] for j in range(len(pair)) if clusters[j] == i])
		ax.scatter(points[:, 0], points[:, 1], s = 5, c = colors[i])
	ax.scatter(centroid[:, 0], centroid[:, 1], marker = '*', s = 150, c = 'black')
	plt.show()




# Two coordinates
def twoCoordinate(w, x, k):
	print("# X, Y Coordinates")
	pair = np.array(list(zip(w, x)))

	accuracy, centroid, clusters= computeClusters(pair, w, x, None, None, k)
	print("Accuracy (k = %d): %.2f%%"%(k, accuracy*100))

	print("# End Two Coordinates\n")

	printScatter(centroid, pair, clusters, k)





# Three coordinates
def threeCoordinate(w, x, y, k):
	print("# X, Y, Speed Coordinates")
	pair = np.array(list(zip(w, x, y)))

	accuracy = computeClusters(pair, w, x, y, None, k)
	print("Accuracy (k = %d): %.2f%%"%(k, accuracy*100))

	print("# End Three Coordinates\n")





# ax, ay, az coordinates
def XYZCoordinate(w, x, y, k):
	print("# AX, AY, AZ Coordinates")
	pair = np.array(list(zip(w, x, y)))

	accuracy = computeClusters(pair, w, x, y, None, k)
	print("Accuracy (k = %d): %.2f%%"%(k, accuracy*100))

	print("# End XYZ Coordinates\n")





# Four coordinates
def fourCoordinate(w, x, y, z, k):
	print("# Pfx_x, Pfx_z, Spin, Speed Coordinates")
	pair = np.array(list(zip(w, x, y, z)))

	accuracy = computeClusters(pair, w, x, y, z, k)
	print("Accuracy (k = %d): %.2f%%"%(k, accuracy*100))

	print("# End Four Coordinates\n")





# Compute Clusters
def computeClusters(pair, w, x, y, z, k):
	w_difference = (np.max(w) - np.min(w)) / k
	x_difference = (np.max(x) - np.min(x)) / k

	# w coordinates of random centroids
	w_centroid = [np.random.randint(np.min(w) + w_difference*i, np.min(w) + w_difference*(i + 1)) for i in range(k)]
	# x coordinates of random centroids
	x_centroid = [np.random.randint(np.min(x) + x_difference*i, np.min(x) + x_difference*(i + 1)) for i in range(k)]

	# Check type
	if y is None and z is None: # Two coordinates
		centroid = np.array(list(zip(w_centroid, x_centroid)), dtype = np.float32)
	elif z is None: # Three coordinates
		y_difference = (np.max(y) - np.min(y)) / k

		# y coordinates of random centroids
		y_centroid = [np.random.randint(np.min(y) + y_difference*i, np.min(y) + y_difference*(i + 1)) for i in range(k)]

		centroid = np.array(list(zip(w_centroid, x_centroid, y_centroid)), dtype = np.float32)
	else: # Four coordinates
		y_difference = (np.max(y) - np.min(y)) / k
		z_difference = (np.max(z) - np.min(z)) / k

		# y coordinates of random centroids
		y_centroid = [np.random.randint(np.min(y) + y_difference*i, np.min(y) + y_difference*(i + 1)) for i in range(k)]
		# z coordinates of random centroids
		z_centroid = [np.random.randint(np.min(z) + z_difference*i, np.min(z) + z_difference*(i + 1)) for i in range(k)]

		centroid = np.array(list(zip(w_centroid, x_centroid, y_centroid, z_centroid)), dtype = np.float32)

	clusters = np.zeros(len(pair))
	print(centroid)

	centroid, clusters = getFinalCentroid(centroid, pair, clusters, k)
	print(centroid)

	if y is None and z is None:
		return computeAccuracy(clusters, pitch_type), centroid, clusters
	else:
		return computeAccuracy(clusters, pitch_type)





# Main Function
if __name__ == '__main__':

	input_data = pd.read_csv('data_noah.csv')

	x = input_data['x'].values
	y = input_data['y'].values
	speed = input_data['speed'].values
	ax = input_data['ax'].values
	ay = input_data['ay'].values
	az = input_data['az'].values
	pfx_x = input_data['pfx_x'].values
	pfx_z = input_data['pfx_z'].values
	spin = input_data['spin'].values
	pitch_type = input_data['pitch_type']

	# number of clusters
	k = 3

	threeCoordinate(x, y, speed, k)

	XYZCoordinate(ax, ay, az, k)

	fourCoordinate(pfx_x, pfx_z, spin, speed, k)

	twoCoordinate(x, y, k)
