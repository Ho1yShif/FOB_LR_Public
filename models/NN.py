"""Logistic Regression implemented as a single-layer neural network in PyTorch"""

# Setup
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

class LogRegNN(nn.Module):
	def __init__(self, n_features, learning_rate=0.01, n_iters=1000):
		super(LogRegNN, self).__init__()
		"""Bundle multiple layers sequentially"""
		self.stack = nn.Sequential(
			nn.Linear(n_features, 1),
			nn.Sigmoid()
		)
		self.n_features = n_features
		self.learning_rate = learning_rate
		self.n_iters = n_iters
	
	def forward(self, x):
		y_predicted = self.stack(x)
		return y_predicted

	"""Fit using Binary Cross Entropy Loss with stochastic gradient descent for optimization"""
	def fit(self, X_train, y_train):
		criterion = nn.BCELoss()
		optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
		
		"""Reshape y data to column vectors; don't need to change y_test vector"""
		y_train = y_train.view(y_train.shape[0], 1)

		for epoch in range(self.n_iters):
			"""Forward pass"""
			y_predicted = self(X_train.float())
			loss = criterion(y_predicted.float(), y_train.float())

			"""Backward pass"""
			loss.backward()

			"""Update gradients"""
			optimizer.step()

			"""zero gradients before next iteration"""
			optimizer.zero_grad()
		
		print('Successfully fit Logistic Regression network to input data.')

	def calc_metrics(self, X_test, y_test):
		with torch.no_grad():
			y_predicted = self(X_test.float())
			y_class = y_predicted.round()
			
		"""Count true positives, true negatives, false positives, and false negatives"""
		self.true_pos = sum([1 for p,a in zip(y_class, y_test) if int(p)==int(a)==1])
		self.false_pos = sum([1 for p,a in zip(y_class, y_test) if int(p)==1 and int(a)==0])
		self.true_neg = sum([1 for p,a in zip(y_class, y_test) if int(p)==int(a)==0])
		self.false_neg = sum([1 for p,a in zip(y_class, y_test) if int(p)==0 and int(a)==1])

		"""Calculate accuracy, precision, recall, and F1-Score; account for ZeroDivisionErrors"""
		self.accuracy = (y_class == y_test).sum() / len(y_test)
		
		try:
			self.precision = self.true_pos / (self.true_pos + self.false_pos)
		except ZeroDivisionError:
			self.precision = 0.0
		
		try:
			self.recall = self.true_pos / (self.true_pos + self.false_neg)
		except ZeroDivisionError:
			self.recall = 0.0
		
		try:
			self.f1 = (self.precision * self.recall) / (self.precision + self.recall)
		except ZeroDivisionError:
			self.f1 = 0.0
		
		metrics = [float(self.accuracy)/100, self.precision, self.recall, self.f1]
		percents = [f'{metric*100:.4}%' for metric in metrics]
		metrics_df = DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'], 'Value': percents})

		return metrics_df
   
	def confusion_matrix(self, X_test, y_test):
		with torch.no_grad():
			y_predicted = self(X_test.float())
			y_class = y_predicted.round()
		
		"""Calculate TP, FP, TN, FN values for confusion matrix"""
		self.true_pos = sum([1 for p,a in zip(y_class, y_test) if int(p)==int(a)==1])
		self.false_pos = sum([1 for p,a in zip(y_class, y_test) if int(p)==1 and int(a)==0])
		self.true_neg = sum([1 for p,a in zip(y_class, y_test) if int(p)==int(a)==0])
		self.false_neg = sum([1 for p,a in zip(y_class, y_test) if int(p)==0 and int(a)==1])
		
		matrix = DataFrame({'Index': ['Predicted', 'True', 'False'],
					'': ['True', self.true_pos, self.false_neg],
					'Actual ': ['False', self.false_pos, self.true_neg]})
		matrix = matrix.set_index('Index')
		
		return matrix

if __name__ == "__main__":
	import os
	import pandas as pd
	from pandas import DataFrame
	import numpy as np
	import matplotlib
	from matplotlib import pyplot as plt
	from sklearn.model_selection import train_test_split
	import torch
	import torch.nn as nn

	# Display entire dataframe
	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', None)

	# Read encoded feature dataframe and separate predictor variables
	features = pd.read_csv('data/processed/features_encoded.csv', index_col=0)
	y = features.pop('class')

	# Split data 80-20
	X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=0)
	n = int(len(X_train.columns))

	# Convert data to Torch objects
	X_train = torch.tensor(X_train.values)
	X_test = torch.tensor(X_test.values)
	y_train = torch.tensor(y_train.values)
	y_test = torch.tensor(y_test.values)

	# Train model
	lrnn = LogRegNN(n, 0.001, 10000)
	lrnn.fit(X_train, y_train)
	
	# View metrics
	print(lrnn.calc_metrics(X_test, y_test))
	print(lrnn.confusion_matrix(X_test, y_test))