import numpy
import sys
from scipy import ndimage
sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
from neural_network import *
from matplotlib.backend_bases import MouseButton
# %matplotlib inline

numpy.set_printoptions(threshold=sys.maxsize)

def interpolate(scaled_input, degree):
	inputs_plus10_img = ndimage.interpolation.rotate(scaled_input.reshape(28,28), degree, cval=0.01, reshape=False)
	return inputs_plus10_img.reshape(784)

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

n.load("24-01-2022_22-47-26")
n.lr = 0.01
# Train

epochs = 1
i = 0

for e in range(epochs):
	for record in training_data_list:
		# print(i / len(training_data_list))
		all_values = record.split(',')
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		original_inputs = inputs
		targets = numpy.zeros(outputNodes) + 0.01
		targets[int(all_values[0])] = 0.99
		inputs_p10 = interpolate(inputs, 5)
		n.train(inputs_p10, targets)
		inputs_m10 = interpolate(original_inputs, -5)
		n.train(inputs_m10, targets)
		# n.train(original_inputs, targets)
		i += 1
		if (i % 1000 == 0):
			print(i)
			# if (i == 5000):
				# break
		pass
	pass

print("done")
# n.load("24-01-2022_21-11-00")
n.save()

# Testing

n.test("mnist_dataset/mnist_test.csv")

# # Showcase

# plt.ion()
# offset = 0

# def on_click(event):
# 	global offset
# 	if event.button is MouseButton.LEFT:
# 		offset += 1
# 		showNumber(wrong_records[offset])
# 	if event.button is MouseButton.RIGHT:
# 		offset -= 1
# 		showNumber(wrong_records[offset])

# def showNumber(wrong_record):
# 	plt.clf()
# 	all_values = wrong_record[0].split(',')
# 	print(wrong_record[1])
# 	print("Neural network:" + str(numpy.argmax(wrong_record[1])))
# 	print("Real value:" + str(all_values[0]))
# 	image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# 	plt.imshow(image_array, cmap='Greys', interpolation='None')
# 	plt.show(block=True)

# plt.connect('button_press_event', on_click)
# showNumber(wrong_records[offset])

