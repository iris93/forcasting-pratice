import numpy as np
from scipy.special import expit

# neural network class definition
class neuralNetwork:

	# initialize the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# set number of nodes in each input, hidden, output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		# link weight matrices, wih and who
		# weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
		# w11 w21
		# w12 w22 etc
		self.wih = np.random.normal(0.0, np.power(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, np.power(self.onodes, -0.5), (self.onodes, self.hnodes))

		# learning rate
		self.lr = learningrate

		# activation_function is the sigmoid function
		self.activation_function = lambda x: expit(x)

		# get the output errors for error analysis
		self.output_errors = 0

	# train the neural network
	def fit(self, inputs_list, targets_list):
		# convert inputs_list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		# FORWARD

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)

		# BACK

		# output layer error is the (target-actual)
		output_errors = targets - final_outputs
		self.output_errors = output_errors

		# hidden layer error is the output_errors, split by weight, recombined at hidden nodes
		hidden_errors = np.dot(self.who.T, output_errors)

		# update the weights for the links between the hidden and output layers
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

		# update the weights for the links between the input and hidden layers
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


	# query the neural network
	def predict(self, inputs_list):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

	# get the final errors
	def final_error(self):
		return self.output_errors
