import numpy
import sys
sys.path.append('../')
from scipy import ndimage
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
from neural_network import *
from matplotlib.backend_bases import MouseButton

inputNodes = 1
hiddenNodes = 10
outputNodes = 2
learningRate = 0.3

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

# Load Data

training_data_file = open("./titanic/train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

print("done")

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

# entrée => 0/1 (male/female)
# sortie => [0.3, 0.7]

def train(training_data_list):
	i = 0
	training_data_list = training_data_list[1:]
	for person in training_data_list:
		all_values = person.split(',')
		sexe = (1, 0)[all_values[5]=="male"]
		survived = all_values[1]
		inputs = (numpy.asfarray(sexe) * 0.99) + 0.01
		inputs = int(sexe)
		targets = int(survived)
		print(all_values[0])
		print("survived: " + str(survived))
		print("sexe: " + str(sexe))
		n.train(inputs, targets)
		i = i + 1
		if (i >= 500):
			break
		pass
	pass

train(training_data_list)

# n.train()

# Testing

# n.test("./titanic/train.csv")

def test():
	# "mnist_dataset/mnist_test.csv"
	test_data_file = open("./titanic/train.csv", 'r')
	test_data_list = test_data_file.readlines()[1:]
	test_data_file.close()
	scoreboard = []
	for record in test_data_list:
		all_values = record.split(',')
		correct_label = int(all_values[1])
		sexe = all_values[5]
		print(correct_label)
		print(sexe)
		pass
		# inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		# outputs = self.query(inputs)
		# label = numpy.argmax(outputs)
	# 	if (label == correct_label):
	# 		scoreboard.append(1)
	# 	else:
	# 		scoreboard.append(0)
	# 		pass
	# 	pass
	# scorecard_array = numpy.asarray(scoreboard)
	# print("Performance = ", scorecard_array.sum() / scorecard_array.size)


# test()