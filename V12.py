# Imports
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import inspect
import random
import pickle
import numpy

class VanillaNetwork(): # the neural network
	def __init__(self, nodeConfig, epochs, saved, batch_size, LR): # Initialize all used variables
		self.Nodes = [i for i in nodeConfig] # Create handy array
		self.Weights = [numpy.random.normal(0.0, pow(self.Nodes[i],-0.5), (self.Nodes[i+1],self.Nodes[i])) for i in range(len(self.Nodes)-1)] # Set random weights efficiently
		self.epochs = epochs # Set epochs, wich will determine how many times the neural network will go over input data 
		self.saved = saved # Set the value wich determines if the neural network should train
		self.batch_size = batch_size # Set batch Size wich determines in how many peaces the input data should be split up
		self.LR = LR # Set learning rate, this determines how severely the weights will change
		self.score = 0 # initialize score wich will be used to calculate the accuracy

	def sigmoid(self,z):
		return 1.0/(1.0+numpy.exp(-z)) # calculate the sigmoid of the value z

	def FeedForward(self, input_list): # Calculate each value of each node in each layer
		Node_Values = [numpy.array(input_list,ndmin=2).T] # initialize array with the values inputs
		for i in range(len(self.Nodes)-1): # cycle through each layer
			Node_Values.append(self.sigmoid(numpy.dot(self.Weights[i], Node_Values[i]))) # Calculate each value of each node in the layer i

		return Node_Values # Return all Values of each node

	def Errors(self, target_list, Node_Values): # Calculates errors of each layer
		errors = [numpy.array(target_list,ndmin=2).T - Node_Values[-1]] # initialize the errors array with the error of the output layer
		for i in range(1,len(self.Nodes)): # cycle through each layer 
			errors.append(numpy.dot(self.Weights[-i].T, errors[i-1])) # Calculate the error of each weight and append to the full errors array

		return errors # Return the output array of each weight

	def UpdateWeights(self, Node_Values, errors, LR): # Updates the weights according to the error of each layer
		for i in range(1,len(self.Nodes)): # cycle through each layer
			self.Weights[-i] += LR * numpy.dot((errors[i-1] *Node_Values[-i] * (1.0 - Node_Values[-i])), numpy.transpose(Node_Values[-i-1])) # Change the weight according to the error of each weight

	def Train(self, input_list, target_list, LR): # This will apply backpropagation and thus train the network
		Node_Values = self.FeedForward(input_list) # Calculate all node values in the network
		errors = self.Errors(target_list, Node_Values) # Calculate the errors of each layer

		self.UpdateWeights(Node_Values, errors, LR) # Call function that changes the weights (backpropagation)

	def Use(self, input_list): # Wil run the neural network and return the output to the user
		out_index = [0,0] # initialize variable wich will contain index and certainty of what the neural network thinks is correct
		for i in range(len((numpy.array(self.FeedForward(input_list)[-1],ndmin=2).T)[0])): # loop through the output list
			if (numpy.array(self.FeedForward(input_list)[-1],ndmin=2).T)[0][i] > out_index[0]: out_index = [(numpy.array(self.FeedForward(input_list)[-1],ndmin=2).T)[0][i],i] # Find the index of highest number (what the neural network thinks the output is)

		print("Result index of input: " + str(out_index[1]) + ". With a certrainty of: " + str(numpy.array(self.FeedForward(input_list)[-1],ndmin=2).T[0][out_index[1]]) + "%.") # Print what the neural network thinks the output is and with what certainty

	def Test(self, input_list, target_list): # This funtion will compare neural network output to the correct output
		input_list, target_list = input_list[0], target_list[0] # The values are at the 0th index
		NN_outputs = numpy.array(self.FeedForward(input_list)[-1],ndmin=2).T # calculate the transposed matrix of the output of the neural network
		TempValue, TempValue1 = [0,0], [0,0] # initialize these values

		for i in range(len(NN_outputs[0])): # loop through the anwer of the neural network
			if NN_outputs[0][i] > TempValue[0]: TempValue = [NN_outputs[0][i], i] # Find the answer the neural network thinks is right

		for i in range(len(target_list)): # loop through the target list
			if target_list[i] > TempValue1[0]: TempValue1 = [NN_outputs[0][i], i] # Find the correct answer in target dataset

		if TempValue[1] == TempValue1[1]: self.score +=1 # Increase score by one if the neural network guessed correctly

	def Get_resultsFromtest(self, index): # This function will print the accuracy and give a small test so the user can verify its accuracy
		input_list, output_list = self.GetData(1, True) # Get the test dataset the True is set to specify you want test data

		for i in range(len(input_list)): # Go through all samples in batch
			self.Test(input_list[i], output_list[i]) # Call test function to compare neural network output vs actual output and saving it in score

		print(str(self.score/len(input_list)*100)+"% of total neural network accuracy." ) # Print the neural network accuracy

		self.Use(input_list[index]) # Prints what the neural network thinks the number at the index specified is
		plt.show(plt.imshow(numpy.asfarray(input_list[index][0]).reshape((28,28)),cm.gray)) # Will show the image at the index specified when the function is called

	def GetData(self, batch_size, Test): # Rewrite just this function if you wan your custom dataset
		if Test: mnist = data_file = open("mnist/mnist_test.csv","r") # Read The test data if Test is set to True when the funtion is called
		else: mnist = data_file = open("mnist/mnist_train.csv","r") # Read the train data if Test is set to False when the function is called

		data_list = data_file.readlines() # Read all lines from training data
		data_file.close() # Close training data

		input_list = [[(numpy.asfarray(data_list[record].split(",")[1:]) / 255 * 0.99) + 0.01 for record in range(0+i*batch_size,batch_size+i*batch_size)] for i in range(int(len(data_list)/batch_size))] # Create a array with batch_size of input samples
		output_list = [] # initialize output list

		for i in range(int(len(data_list)/batch_size)): # Loop for batch size times so we have batch size samples
			TempOutputs = [] # Reset TempOutputs to nothing
			for record in range(0+i*batch_size,batch_size+i*batch_size): # Got through every row of pixels of every sample
				targets = numpy.zeros(self.Nodes[-1]) + 0.01 # Set stanard targets to 0.01
				targets[int(data_list[record].split(",")[0])] = 0.99  # Set the correct targets to 0.99
				TempOutputs.append(targets) # Add this sample to the batch
			output_list.append(TempOutputs) # Add batch to output

		return input_list, output_list # Return list

	def WeightSaveLoad(self): # Save or Load the weights
		f = open("Weights/"+str(str(inspect.getfile(inspect.currentframe())).split(".")[0])+".txt","r+b") #Open the weights file
		if 1==self.saved: self.Weights = pickle.load(f) # If the weights are already saved (is set in self.saved variable) it will load them as the weights 
		else: pickle.dump(self.Weights, f) # The neural network is done training and will save the weights into the file
		f.close() # Close the file

	def SelfMadeImages(self): # This is the function you call if you made your own image or if you want to use it yourself (mnist only)
		im = numpy.asfarray(mpimg.imread('mnist/TestImage.png')) # Read your image
		self.Use(im.reshape(784)) # Get the answer from the neural network
		plt.show(plt.imshow(im, cm.gray)) # Show the image on screen

	def Main(self): # Main Function
		if 1!=self.saved: # Check if you should train or simply load
			input_list, output_list = self.GetData(self.batch_size, False) # Get data, The false is so it knows we're looking to get train data
			print("starting training") # Handy print to show when it is done getting the training data
			for e in range(self.epochs): # Cycle through the amount of epochs
				print("\n####### epoch " + str(e) + " out of " + str(epochs)+ ". #######\n") # Print when a new epoch starts
				for i in range(len(input_list)): # Cycle through all the batches
					if i % 5 == 0: print("batch " + str(i) + " out of " + str(len(input_list)))	# Print the current batch every 5 batches for progress report
					self.Train(input_list[i], output_list[i],self.LR) # Train the neural network on this batch

		# Load or save the weights
		self.WeightSaveLoad() # !!!!!!!!!!!!!! Note: File V12.txt must exist in the Weights/ directory !!!!!!!!!!!!!!

		# Call function to test the accuracy
		self.Get_resultsFromtest(4444) # The number is the index of a image in the test dataset. 4852 is also a nice index to test

		# In case you have your own image you want to test uncomment this line:
		#self.SelfMadeImages()

		

Config = [28*28, 200, 10] # The Neuron configuration [n_input_nodes, n_hiddenlayer1_nodes, ...., n_output_nodes]
LearningRate = 0.01 # The severity of wich the weights will change each step
batch_size = 500 # The batch size (increasing/decreasing might result in a performance boost)
epochs = 5 # The amount of times the neural network will go over the training data
saved = 0 # 0 means you want the neural network to train, 1 means you have saved weights. Make sure your config matches your saved weights file!

NN = VanillaNetwork(Config,epochs, saved, batch_size, LearningRate) # Set neural network object to NN
NN.Main() # execute the Main function of neural network