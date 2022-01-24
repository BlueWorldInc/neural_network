import numpy
import scipy.special
from shutil import copy2
from datetime import datetime

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

	def save(self, description = ""):
		weigthsFile = open('./weigths/weigths.txt', 'w')
		weigthsFile.write(str((self.wih.tolist())))
		weigthsFile.write(str((self.who.tolist())))
		weigthsFile.write("Description: " + description)
		weigthsFile.close()
		now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
		copy2('./weigths/weigths.txt', './weigths/weigths'+now+'.txt')
		pass

	def load(self, date):
		weigthsFile = open('./weigths/weigths'+date+'.txt')
		weigthsFileContent = weigthsFile.read()
		self.wih = numpy.array(eval(weigthsFileContent.split(']]')[0] + ']]'))
		self.who = numpy.array(eval(weigthsFileContent.split(']]')[1] + ']]'))
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

		return final_outputs

	def test(self, test_data_file_path):
		# "mnist_dataset/mnist_test.csv"
		test_data_file = open(test_data_file_path, 'r')
		test_data_list = test_data_file.readlines()
		test_data_file.close()
		scoreboard = []
		wrong_records = [] 

		for record in test_data_list:
			all_values = record.split(',')
			correct_label = int(all_values[0])
			inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			outputs = self.query(inputs)
			label = numpy.argmax(outputs)
			if (label == correct_label):
				scoreboard.append(1)
			else:
				scoreboard.append(0)
				wrong_records.append([record, outputs])
				pass
			pass

		scorecard_array = numpy.asarray(scoreboard)
		print("Performance = ", scorecard_array.sum() / scorecard_array.size)