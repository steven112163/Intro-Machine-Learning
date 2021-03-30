import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree, preprocessing
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
	print("Accuracy: %.3f%%"%(accuracy_score(target, predict)*100)) # Accuracy (%)
	print("Precision: ", precision_score(target, predict, average = None)) # Precision
	print("Recall: ", recall_score(target, predict, average = None)) # Recall
# End





# Compute K-Fold
def computeKFold(data, target, model):
	skf = StratifiedKFold(n_splits = 40, shuffle = True)
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
	for i in range(22):
		clf = tree.DecisionTreeClassifier()
		n_features = np.random.randint(6, size = np.random.randint(low = 4, high = 7)) # Generate random features
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





if __name__ == '__main__':
	# Load the data
	raw_data = pd.read_csv("googleplaystore.csv")

	# Drop the data which contains "NaN" in its row
	pre_data = raw_data.fillna(0)

	# Drop useless columns
	input_data = pre_data.drop(['App', 'Last Updated', 'Current Ver', 'Size', 'Genres', 'Price'], axis = 1)
	
	# Show information
	print("# Information of Google Data")
	print(input_data.describe())
	print("# End IOGD")
	print("\n\n\n")

	# Transfer the data to make it readable for DecisionTreeClassifier
	le = preprocessing.LabelEncoder()
	input_data['Category'] = le.fit_transform(input_data['Category'].astype(str))
	input_data['Type'] = le.fit_transform(input_data['Type'].astype(str))
	input_data['Content Rating'] = le.fit_transform(input_data['Content Rating'].astype(str))
	input_data['Android Ver'] = le.fit_transform(input_data['Android Ver'].astype(str))
	input_data.replace(['0', '0+', '1+', '5+', '10+', '50+'], 0, inplace = True)
	input_data.replace(['100+', '500+', '1,000+', '5,000+', '10,000+'], 1, inplace = True)
	input_data.replace(['50,000+', '100,000+', '500,000+', '1,000,000+', '5,000,000+'], 2, inplace = True)
	input_data.replace(['10,000,000+', '50,000,000+', '100,000,000+', '500,000,000+', '1,000,000,000+'], 3, inplace = True)

	#get "Installs" column and set it as target
	target = input_data['Installs']
	data = input_data.drop("Installs", axis = 1)

	# Get and plot decision tree
	GenerateTree(data, target)
	
	# Get and plot random forest
	GenerateForest(data, target)
