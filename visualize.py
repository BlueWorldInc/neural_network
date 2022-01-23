from main import *
import sys

def wait():
	while plt.waitforbuttonpress():
		break

def showNumber(number_line):
	plt.clf()
	all_values = number_line.split(',')
	image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
	plt.imshow(image_array, cmap='Greys', interpolation='None')
	plt.show()
	wait()

#Init
# inputNodes = 784
# hiddenNodes = 200
# outputNodes = 10
# learningRate = 0.1

# n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
number = 0
if (len(sys.argv) > 1):
	number = int(sys.argv[1])

plt.ion()

#Train
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# all_values = training_data_list[number].split(',')
# print(all_values[0])
# image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# plt.imshow(image_array, cmap='Greys', interpolation='None')

# plt.draw(block=True)
# plt.show()
# press = plt.waitforbuttonpress(0.1)

# wait()

showNumber(training_data_list[number])
showNumber(training_data_list[number+1])

# plt.clf()

# all_values = training_data_list[number+1].split(',')
# image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# plt.imshow(image_array, cmap='Greys', interpolation='None')
# plt.show()
# plt.pause(1)

