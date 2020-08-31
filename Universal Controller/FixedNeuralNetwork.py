import numpy as np

'''a basic implementation of a fixed Neural Network 
Network is designed with one layer of hidden neurons, set to be twice the number of input neurons
'''


class FixedNeuralNetwork():
    def __init__(self, inputs, outputs = 2):
        
        self.inputs = inputs
        self.hidden_neurons = self.inputs*2
        #self.hidden_neurons = self.inputs
        self.output_neurons = outputs
        
        self.weights_one = np.zeros((self.hidden_neurons, self.inputs), dtype=float)
        self.weights_two = np.zeros((self.output_neurons, self.hidden_neurons), dtype=float)


        self.biases_one = np.zeros(self.hidden_neurons, dtype=float)
        self.biases_two = np.zeros(self.output_neurons, dtype=float)
        
        #calculates the size of the necessary algorithm solutions to assign all weights and biases
        self.solution_size = ((self.inputs*self.hidden_neurons)) + self.hidden_neurons + (self.hidden_neurons * self.output_neurons) + self.output_neurons

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
    
    def forwardPass(self, inputs):
        #forward pass - preactivation(preA) and activation(H) - standard NN notation
        #create arrays to hold upcoming preactivation totals acting on the neurons
        preA = np.zeros((self.hidden_neurons, self.inputs))
        
        #each output neuron has x number of inputs acting on it
        preA2 = np.zeros((self.output_neurons, self.hidden_neurons))
        
        #array to hold hidden neuron activation totals
        H = np.zeros(self.hidden_neurons)
        
        #array for final outputs from output neurons
        finalOutputs = np.zeros(self.output_neurons, dtype=float)
        
        #calculate the output of each neuron in the hidden layer
        for n in range(self.hidden_neurons):
            for j in range(self.inputs):
                preA[n][j] = inputs[j] * self.weights_one[n][j]
    
            hOut = self.sigmoid(np.sum(preA[n]) + self.biases_one[n])
            H[n] = hOut
        
        #calculate the output of each neuron in the output layer
        for n in range(self.output_neurons):
            for j in range(len(H)):
                preA2 [n][j] = H[j] * self.weights_two[n][j]

            finalOutputs[n] = self.sigmoid(np.sum(preA2[n]) + self.biases_two[n])
            
        return finalOutputs

    #decode an individual passed from an evolutionary algorithm
    def decodeEA(self, individual):
        #overall index keeps track of decoding progress
        index = 0
        
        #decode the hidden neuron weights and biases
        for i in range(self.hidden_neurons):
            for j in range(self.inputs):
                self.weights_one[i][j] = individual[index]
                index +=1
            self.biases_one[i] = individual[index]
            index += 1
        
        #decode the weights and biases for the output neurons
        for k in range(self.output_neurons):
            for l in range(self.hidden_neurons):
                self.weights_two[k][l] = individual[index]
                index += 1
                
            self.biases_two[k] = individual[index]
            index += 1