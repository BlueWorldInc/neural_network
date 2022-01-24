from main import *
from matplotlib.backend_bases import MouseButton
import sys

waiting = True
offset = 0

def waitUntilClick():
	while plt.waitforbuttonpress():
		break
	# while waiting:
		# pass
		# break
		# plt.pause(0.1)
		# if ()

def on_click(event):
	global offset
	if event.button is MouseButton.LEFT:
	# 	# print('disconnecting callback')
	# 	# waiting = not waiting
		offset += 1
		showNumber(training_data_list[offset])
	if event.button is MouseButton.RIGHT:
	# 	# print('disconnecting callback')
	# 	# waiting = not waiting
		offset -= 1
		showNumber(training_data_list[offset])

def showNumber(number_line):
	global offset
	current_offset = offset
	# waiting = True
	plt.clf()
	all_values = number_line.split(',')
	image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
	plt.imshow(image_array, cmap='Greys', interpolation='None')
	plt.show(block=True)
	# while current_offset == offset:
		# plt.pause(0.1)
	# else:
		# showNumber(training_data_list[offset])
	# waitUntilClick()
	# plt.pause(1)
	# print("1")

number = 0
if (len(sys.argv) > 1):
	number = int(sys.argv[1])

plt.ion()

training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
plt.connect('button_press_event', on_click)
# waitUntilClick()
showNumber(training_data_list[offset])
# for numb in range(2):
	# showNumber(training_data_list[numb])


# montrer une image
# attendre pour un click
# lors du click afficher une autre image (maj)
# si click gauche ca augmente, si click gauche ca diminue
# si click fleche ca augmente, si fleche gauche ca diminue
# si on click sur fermer ca stop le programme