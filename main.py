import numpy
import sys
import scipy.special
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
# %matplotlib inline

numpy.set_printoptions(threshold=sys.maxsize)

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

	def save(self):
		weigthsFile = open('./weigths/weigths.txt', 'w')
		weigthsFile.write(str((n.wih.tolist())))
		weigthsFile.write(str((n.who.tolist())))
		pass

	def load(self):
		weigthsFile = open('./weigths/weigths.txt')
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

#Init

inputNodes = 784
hiddenNodes = 200
outputNodes = 10
learningRate = 0.1

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

# Load Data

training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Show Data

# all_values = training_data_list[1].split(',')
# print(all_values[0])
# image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# plt.imshow(image_array, cmap='Greys', interpolation='None')
# plt.show()

# Train

# epochs = 1
# i = 0

# for e in range(epochs):
# 	for record in training_data_list:
# 		# print(i / len(training_data_list))
# 		all_values = record.split(',')
# 		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# 		targets = numpy.zeros(outputNodes) + 0.01
# 		targets[int(all_values[0])] = 0.99
# 		n.train(inputs, targets)
# 		i += 1
# 		if (i % 1000 == 0):
# 			print(i)
# 		pass
# 	pass

print("done")
n.load()

#Testing

test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scoreboard = []
wrong_records = [] 

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
		wrong_records.append([record, outputs])
		# print(all_values[0])
		# image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
		# plt.imshow(image_array, cmap='Greys', interpolation='None')
		# plt.show()
		pass
	pass

scorecard_array = numpy.asarray(scoreboard)
# print(scoreboard)
print("Performance = ", scorecard_array.sum() / scorecard_array.size)

# Showcase

plt.ion()
offset = 0

def on_click(event):
	global offset
	if event.button is MouseButton.LEFT:
		offset += 1
		showNumber(wrong_records[offset])
	if event.button is MouseButton.RIGHT:
		offset -= 1
		showNumber(wrong_records[offset])

def showNumber(wrong_record):
	plt.clf()
	all_values = wrong_record[0].split(',')
	print(wrong_record[1])
	print("Neural network:" + str(numpy.argmax(wrong_record[1])))
	print("Real value:" + str(all_values[0]))
	image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
	plt.imshow(image_array, cmap='Greys', interpolation='None')
	plt.show(block=True)

plt.connect('button_press_event', on_click)
showNumber(wrong_records[offset])

