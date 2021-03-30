import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter





# Use "$ dot -Tpng <name>.dot -o <name>.png" to convert dot into png





# Get most common element
def mostCommon(lst):
	data = Counter(lst)
	return max(lst, key = data.get)
# End





# Compute Resub
def computeResub(data, target, model):
	if model == 'tree':
		predict = getTreePrediction(data, data, target) # Get tree prediction
	else:
		predict = getForestPrediction(data, data, target) # Get forest prediction

	print(confusion_matrix(list(target), list(predict))) # Confusion matrix
	print("Accuracy: %.3f%%"%(accuracy_score(target, predict)*100)) # Accuracy
	print("Precision: ", precision_score(target, predict, average = None)) # Precision
	print("Recall: ", recall_score(target, predict, average = None)) # Recall
# End





# Compute K-Fold
def computeKFold(data, target, model):
	skf = StratifiedKFold(n_splits = 5, shuffle = True)
	allMatrix = []
	allAccuracy = []

	for train_index, test_index in skf.split(data, target):
		train_data, test_data = data.iloc[train_index, :], data.iloc[test_index, :] # Split data
		train_target, test_target = target[train_index], target[test_index] # Split target

		if model == 'tree': # Tree
			predict = getTreePrediction(train_data, test_data, train_target) # Get tree prediction
		else: # Random Forest
			predict = getForestPrediction(train_data, test_data, train_target) # Get random forest prediction

		matrix, accuracy = getResults(test_target, predict) # Get results
		allMatrix.append(matrix)
		allAccuracy.append(accuracy)
	
	printResults(allMatrix, allAccuracy) # Print results
# End





# Get tree's prediction
def getTreePrediction(train, test, target):
	clf = tree.DecisionTreeClassifier()
	clf.fit(train, target) # Generate the tree
	predict = clf.predict(test) # Generate prediction
	return predict
# End





# Get forests' prediction
def getForestPrediction(train, test, target):
	predict = []
	for i in range(11):
		clf = tree.DecisionTreeClassifier()
		n_features = np.random.randint(4, size = np.random.randint(low = 2, high = 5)) # Generate random features
		clf.fit(train.iloc[:, n_features], target) # Generate the tree
		predict.append(clf.predict(test.iloc[:, n_features]))
	return getFinalPrediction(predict)
# End





# Get final prediciton according to the forest
def getFinalPrediction(predict):
	final_predict = []
	for col in range(len(predict[0])): # Generate prediction according to the forest
		found_target = []
		for row in range(len(predict)):
			found_target.append(predict[row][col])
		final_predict.append(mostCommon(found_target))
	return final_predict
# End





# Print results
def printResults(matrix, accuracy):
	sumMatrix = matrix[len(matrix) - 1]
	for i in range(len(matrix)-1):
		sumMatrix += matrix[i]
	print(sumMatrix) # Confusion matrix

	print("Accuracy: %.3f%%"%(np.mean(accuracy))) # Accuracy (%)

	TP = np.diag(sumMatrix)
	TP = [x*1.0 for x in TP]
	FP = np.sum(sumMatrix, axis=0) - TP
	FN = np.sum(sumMatrix, axis=1) - TP
	print("Precision: ", TP / (TP+FP)) # Precision
	print("Recall: ", TP / (TP+FN)) # Recall
# End





# Get results
def getResults(target, predict):
	return confusion_matrix(list(target), list(predict)), accuracy_score(target, predict)*100
# End




# Decision Tree
def GenerateTree(data, target):
	print("# Decision Tree\n")

	# Resubstitution Validation
	print("** Resubstitution Validation")
	computeResub(data, target, 'tree')
	print("** End RV\n")
	# End Resubstitution Validation

	# K-Fold Cross Validation
	print("** K-Fold Cross Validation")
	computeKFold(data, target, 'tree')
	print("** End K-Fold\n")
	# End K-Fold Cross Validation

	print("# End DT\n\n\n")
# End Decision Tree





# Forest
def GenerateForest(data, target):
	print("# Random Forest\n")

	# Resubstitution Validation
	print("** Resubstitution Validation")
	computeResub(data, target, 'forest')
	print("** End RV\n")
	# End Resubstitution Validation

	# K-Fold Cross Validation
	print("** K-Fold Cross Validation")
	computeKFold(data, target, 'forest')
	print("** End K-Fold\n")
	# End K-fold Cross Validation

	print("# End RF")
# End Forest




# Plot box plot
def plotIrisBoxPlot(iris_data):
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 10))

	ax1.set_ylabel("Centimeters")
	ax1.set_title("Box Plot of Four Features")
	sns.boxplot(data = iris_data, width = 0.5, fliersize = 5, ax = ax1)

	ax2.set_title("Iris-setosa")
	sns.boxplot(data = iris_data.iloc[0:49, 0:4], width = 0.5, fliersize = 5, ax = ax2)

	ax3.set_xlabel("Features")
	ax3.set_ylabel("Centimeters")
	ax3.set_title("Iris-versicolor")
	sns.boxplot(data = iris_data.iloc[50:99, 0:4], width = 0.5, fliersize = 5, ax = ax3)

	ax4.set_xlabel("Features")
	ax4.set_title("Iris-virginica")
	sns.boxplot(data = iris_data.iloc[100:149, 0:4], width = 0.5, fliersize = 5, ax = ax4)

	fig.tight_layout()
	plt.show()
# End





if __name__ == '__main__':
	# Load data
	input_data = pd.read_csv("iris.data", names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

	# Show information
	print("# Information of Iris Data")
	print(input_data.describe())
	print("# End IOID")
	print("\n\n\n")

	# Preprocess data
	input_data.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], inplace = True)

	# Process data
	iris_data = input_data.drop("class", axis = 1)
	iris_target = input_data['class']

	# Get and plot decision tree
	GenerateTree(iris_data, iris_target)
	
	# Get and plot random forest
	GenerateForest(iris_data, iris_target)

	# Plot Iris data
	plotIrisBoxPlot(iris_data)
