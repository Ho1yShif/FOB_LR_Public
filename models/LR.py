# Setup
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt

class ScratchLogisticRegression:

	def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
		self.learning_rate = float(learning_rate)
		self.loss_history = []
		self.n_iters = n_iters
		self.weights = None
		self.bias =  None
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

	def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> str:
		self.fit = 'fit'
		np.random.seed(0)
		
		"""Store features of input data"""
		self.feature_names_in_ = list(X.columns)
		self.n_features_in_ = len(self.feature_names_in_)
		
		"""Initialize weights and biases to random and zero, respectively"""
		n_rows, m_features = X.shape
		self.weights = np.random.random(m_features)
		self.bias = 0

		"""Define norm and initialize list to track loss function at each iteration"""
		norm = 1 / n_rows
		loss_history = []

		for _ in range(self.n_iters):		
			"""Build fit method's version of linear model as a dot product of the X-vector and the model weights; add the bias"""
			linear_model = np.dot(X, self.weights) + self.bias
			y_predicted = self._sigmoid(linear_model)
			y_predicted, y = y_predicted.astype(float), y.astype(float) # Force float type

			"Compute log_loss (binary cross-entropy loss) function"
			class1_error = y * np.log(y_predicted + np.finfo(float).eps) # Add epsilon term
			class0_error = (1-y) * np.log(1 - y_predicted + np.finfo(float).eps)
			log_loss = -norm * np.sum(class1_error + class0_error) 
			loss_history.append(log_loss)
			
			"""Gradient Descent: Calculate partial derivatives
			of the cost function with respect to the weights and biases"""
			residuals = y_predicted - y
			dw = norm * np.dot(X.T, residuals)
			db = norm * np.sum(residuals)
			
			"""Update weights and biases based on the learning rate and derivatives"""
			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db
		self.loss_history = loss_history
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
		self.is_predicted = 'predicted'

		if self.is_fit is None:
			self.fit(X)
		
		probabilities = self.predict_proba(X)
		self.predicted_classes = [1 if prob > 0.5 else 0 for prob in probabilities]
		return self.predicted_classes

	def baseline(self, X: pd.DataFrame) -> np.ndarray:
		"""Build naïve baseline model that outputs random ones and zeros as class predictions"""
		rows = X.shape[0]
		self.baseline_predictions = np.random.randint(low=0, high=2, size=rows)
		return self.baseline_predictions
	
	def metrics(self, X: pd.DataFrame, actuals: pd.DataFrame, predictions: list | pd.DataFrame | np.ndarray = None) -> pd.DataFrame:
		if predictions is None:
			if self.is_predicted is None:
				self.predict(X)
			predictions = self.predicted_classes

		"""Count true positives, true negatives, false positives, and false negatives"""
		self.true_pos, self.true_neg, self.false_pos, self.false_neg = 0, 0, 0, 0
		
		for p_label, a_label in zip(predictions, actuals):
			p_label, a_label = int(p_label), int(a_label)

			if p_label == 1 and a_label == 1:
				self.true_pos += 1
			if p_label == 0 and a_label == 0:
				self.true_neg += 1
			if p_label == 1 and a_label == 0:
				self.false_pos += 1
			if p_label == 0 and a_label == 1:
				self.false_neg += 1
		
		"""Calculate accuracy, precision, recall, and F1-Score; account for ZeroDivisionErrors"""
		self.accuracy = round((predictions == actuals).sum() / len(actuals), 2)
		
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

	def confusion_matrix(self, actuals: pd.DataFrame) -> pd.DataFrame:
		"""Display counts of the model's true positives, true negatives, false positives, and false negatives"""
		self.metrics(actuals)
		matrix = DataFrame({'Index': ['Predicted', 'True', 'False'],
						'': ['True', self.true_pos, self.false_neg],
						'Actual ': ['False', self.false_pos, self.true_neg]})
		self.matrix = matrix.set_index('Index')
		return self.matrix
	
	def plot_loss(self)-> str:
		final = f'Final Loss: {self.loss_history[-1]:.2f}'
		
		"""Set up Loss figure with title and axis labels"""
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_title("Loss")
		ax.set_xlabel('Epochs')
		ax.set_ylabel('Log Loss')
		
		"""Plot loss history over iterations"""
		loss_plot = plt.plot(np.arange(self.n_iters), self.loss_history)
		plt.show()
		return final

# Test LR class
if __name__ == '__main__':
	# lr = ScratchLogisticRegression(0.00001, 10000)
	# arr1 = np.ndarray([8, 7, 6],[8, 7, 6],[8, 7, 6])
	# arr2 = np.ndarray([1, 1, 1]).flatten()
	# lr.fit(arr1, arr2)
	# print(lr.metrics(arr1, arr2))
	# lr.plot_loss()
	# print(lr._sigmoid(DataFrame([-1, 0, 1])))
	# print(lr.update_params(.03))