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
		pass

	# train the neural entwork
	def train():
		pass

	# query the neural network
	def query():
		pass

inputNodes = 3
hiddenNodes = 3
outputNodes = 3
learningRate = 0.4

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)