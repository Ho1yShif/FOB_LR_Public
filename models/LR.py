"""Logistic Regression implemented from scratch in Python and NumPy"""

# Setup
import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt

class ScratchLogisticRegression:

	def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
		"""Model hyperparameters"""
		self.learning_rate = float(learning_rate)
		self.n_iters = n_iters
		
		"""Properties related to cost function"""
		self.weights = None
		self.bias =  None
		self.train_loss = []

		"""Checks to see whether model has already been fit or used to predict classes"""
		self.is_fit = None
		self.is_predicted = None
	
	def __repr__(self) -> str:
		return f'<ScratchLogisticRegression(learning_rate={self.learning_rate}, n_iters={self.n_iters})>'

	def __str__(self) -> str:
		return \
		f"""ScratchLogisticRegression object
		self.learning_rate = {self.learning_rate}
		self.n_iters = {self.n_iters}
		self.weights = {self.weights}
		self.bias =  {self.bias}"""

	def _sigmoid(self, x: pd.DataFrame) -> np.ndarray:
		"""Define stable sigmoid function which will be used in the gradient descent loop"""
		x = x.astype(float) # Convert types
		stable_sig = np.where(x < 0, \
			# If x < 0
			np.exp(x)/(1 + np.exp(x)), \
			# If x >= 0
			1/(1 + np.exp(-x)))

		return stable_sig
	
	def update_params(self, new_learning_rate: float = None, new_n_iters: int = None) -> str:
		"""Account for missing arguments"""
		if new_learning_rate is None:
			new_learning_rate = self.learning_rate
		if new_n_iters is None:
			new_n_iters = self.n_iters
			
		self.learning_rate = new_learning_rate
		self.n_iters = new_n_iters
		return f'learning rate: {self.learning_rate}, number of iterations: {self.n_iters}'

	def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
	X_test: pd.DataFrame, y_test: pd.DataFrame) -> str:
		self.is_fit = True
		np.random.seed(0)
		
		"""Store features of training data"""
		self.feature_names_in_ = list(X_train.columns)
		self.n_features_in_ = len(self.feature_names_in_)
		
		"""Initialize weights and bias to random and zero, respectively"""
		n_rows, m_features = X_train.shape
		self.weights = np.random.random(m_features)
		self.bias = 0

		"""Initialize lists to track both training and test error at each epoch"""
		train_loss, test_loss, dw_history, db_history = [], [], [], []
		Xs, ys = [X_train, X_test], [y_train, y_test]
		norms = [(1 / len(X_train)), (1 / len(X_test))]
		
		for is_training, (X, y, norm) in enumerate(zip(Xs, ys, norms)):
			for _ in range(self.n_iters):		
				"""Build fit method's version of linear model as a dot product of the X-vector and the model weights; add the bias"""
				linear_model = np.dot(X, self.weights) + self.bias
				y_predicted = self._sigmoid(linear_model)
				y_predicted, y = y_predicted.astype(float), y.astype(float) # Force float type

				"""Compute log_loss (binary cross-entropy loss) function"""
				class1_error = y * np.log(y_predicted + np.finfo(float).eps) # Add epsilon term to prevent errors
				class0_error = (1-y) * np.log(1 - y_predicted + np.finfo(float).eps)
				log_loss = -norm * np.sum(class1_error + class0_error) 
				
				is_training = int(is_training)
				"""Case 0 corresponds to training error tracking; case 1 corresponds to test error"""
				match is_training:
					case 0:
						train_loss.append(log_loss)
					case 1:
						test_loss.append(log_loss)
				
				"""Only update params when we're using training data (when is_training==0)"""
				if is_training == 0:
					"""Gradient Descent: Calculate partial derivatives
					of the cost function with respect to the weights and biases"""
					residuals = y_predicted - y

					dw = norm * np.dot(X.T, residuals)
					db = norm * np.sum(residuals)
					
					dw_history.append(dw)
					db_history.append(db)
				
					"""Update weights and bias based on the learning rate and derivatives"""
					self.weights -= self.learning_rate * dw
					self.bias -= self.learning_rate * db
		
		self.train_loss, self.test_loss, self.dw_history, self.db_history = train_loss, test_loss, dw_history, db_history
		return 'Successfully fit Logistic Regression model to input data.'

	def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
		"""Build predict method's version of linear model as a dot product of the X-vector and updated model weights; add the updated bias
			Run sigmoid function to convert predictions to probabilities of Class 1 Membership"""
		if self.is_fit is None:
			self.fit(X)
		
		pred_linear_model = np.dot(X, self.weights) + self.bias
		return self._sigmoid(pred_linear_model)
	
	def predict(self, X: pd.DataFrame) -> list:
		"""Output predicted classes for each data point"""
		self.is_predicted = True

		probabilities = self.predict_proba(X)
		self.predicted_classes = [1 if prob > 0.5 else 0 for prob in probabilities]
		return self.predicted_classes

	def baseline(self, X: pd.DataFrame) -> np.ndarray:
		"""Build naïve baseline model that outputs random ones and zeros as class predictions"""
		rows = X.shape[0]
		self.baseline_predictions = np.random.randint(low=0, high=2, size=rows)
		return self.baseline_predictions
	
	def metrics(self, X: pd.DataFrame | np.ndarray | list, actuals: pd.DataFrame, predictions: pd.DataFrame | np.ndarray | list = None) -> pd.DataFrame:
		if predictions is None:
			if self.is_predicted is None:
				self.predict(X)
			predictions = self.predicted_classes

		"""Count true positives, true negatives, false positives, and false negatives"""
		self.true_pos, self.true_neg, self.false_pos, self.false_neg = 0, 0, 0, 0
		
		self.true_pos = sum([1 for p,a in zip(predictions, actuals) if int(p)==int(a)==1])
		self.false_pos = sum([1 for p,a in zip(predictions, actuals) if int(p)==1 and int(a)==0])
		self.true_neg = sum([1 for p,a in zip(predictions, actuals) if int(p)==int(a)==0])
		self.false_neg = sum([1 for p,a in zip(predictions, actuals) if int(p)==0 and int(a)==1])

		"""Calculate accuracy, precision, recall, and F1-Score; account for ZeroDivisionErrors"""
		self.accuracy = (predictions == actuals).sum() / len(actuals)
		
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
		
		metrics = [self.accuracy, self.precision, self.recall, self.f1]
		percents = [str(round(metric*100, 2))+'%' for metric in metrics]
		metrics_df = DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'], 'Value': percents})

		return metrics_df

	def confusion_matrix(self, X, actuals: pd.DataFrame) -> pd.DataFrame:
		"""Display counts of the model's true positives, true negatives, false positives, and false negatives"""
		self.matrix = DataFrame({'Index': ['Predicted', 'True', 'False'],
					'': ['True', self.true_pos, self.false_neg],
					'Actual ': ['False', self.false_pos, self.true_neg]})
		
		self.metrics(X, actuals)

		return self.matrix
	
	def plot_loss(self, graph: str = 'both') -> str:
		"""Note final loss values and max loss"""
		final_train = self.train_loss[-1]
		final_test = self.test_loss[-1]

		"""Figure setup with subplots"""
		fig, ax = plt.subplots()
		plt.title('Model Performance')
		plt.xlabel('Epochs')
		plt.ylabel('Log Loss')

		"""Choose whether to graph only training error, only test error, or both (all options show legend)"""
		match graph:
			case 'train':
				max_loss = math.floor(max(self.train_loss)) + 0.2
				ax.plot(self.train_loss, color = 'blue', label = 'Training Error')
				ax.legend(loc='upper center')

			case 'test':
				max_loss = math.floor(max(self.test_loss)) + 0.2
				ax.plot(self.test_loss, color = 'red', label = 'Test Error')
				ax.legend(loc='upper center')
			
			case 'both':
				max_loss = math.floor(max(max(self.train_loss), max(self.test_loss))) + 0.2
				ax.plot(self.train_loss, color = 'blue', label = 'Training Error')
				ax.plot(self.test_loss, color = 'red', label = 'Test Error')
				ax.legend(loc='upper center')
		
		plt.show()
		return f'Final Training Error: {final_train:.2f}, Final Test Error: {final_test:.2f}'

# Test LR class
if __name__ == '__main__':
	import os
	import pandas as pd
	from pandas import DataFrame
	import numpy as np
	import matplotlib
	from matplotlib import pyplot as plt
	from sklearn.model_selection import train_test_split

	# Display entire dataframe
	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', None)

	# Read encoded feature dataframe and separate predictor variables
	features = pd.read_csv('data/processed/features_encoded.csv', index_col=0)
	y = features.pop('class')

	# Split data 80-20
	X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=0)
	
	# Fit logistic regression model
	lr = ScratchLogisticRegression()
	lr.fit(X_train, y_train, X_test, y_test)
	predictions = lr.predict(X_test)
	# lr.baseline(X_test)
	
	# Outputs
	# print(lr.plot_loss())
	print(lr.metrics(X_test, y_test, predictions))
	print(lr.confusion_matrix(X_test, y_test))
	# print(lr.dw_history[-50])
	print()