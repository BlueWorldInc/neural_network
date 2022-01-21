import numpy
import scipy.special
import matplotlib.pyplot
# %matplotlib inline

# neural network class definition
class neuralNetwork:

	# initialise the neural network
	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		# set number of nodes in each input, hidden and output layers
		self.inodes = inputNodes
		self.hnodes = hiddenNodes
		self.onodes = outputNodes
		
		# set learning rate
		self.lr = learningRate

		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		self.activation_function = lambda x: scipy.special.expit(x)

		pass

	# train the neural entwork
	def train(self, inputs_list, targets_list):
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T
		
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		outputs_errors = targets - final_outputs
		hidden_errors = numpy.dot(self.who.T, outputs_errors)

		self.who += self.lr * numpy.dot((outputs_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
		pass

	# query the neural network
	def query(self, inputs_list):
		inputs = numpy.array(inputs_list, ndmin=2).T
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		pass

inputNodes = 784
hiddenNodes = 200
outputNodes = 10
learningRate = 0.2

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 1

for e in range(epochs):
	for record in training_data_list:
		all_values = record.split(',')
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		targets = numpy.zeros(outputNodes) + 0.01
		targets[int(all_values[0])] = 0.99
		n.train(inputs, targets)
		pass
	pass

test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scoreboard = []

for record in test_data_list:
	all_values = record.split(',')
	correct_label = int(all_values[0])
	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	outputs = n.query(inputs)
	label = numpy.argmax(outputs)
	if (label == correct_label):
		scoreboard.append(1)
	else:
		scoreboard.append(0)
		pass
	pass

scorecard_array = numpy.asarray(scoreboard)
print("Performance = ", scorecard_array.sum() / scorecard_array.size)

