import numpy as np

'''a basic implementation of a fixed Neural Network 
Network is designed with one layer of hidden neurons, set to be twice the number of input neurons
'''


class FixedNeuralNetwork():
    def __init__(self, inputs, outputs = 2):
        
        self.inputs = inputs
        self.hiddenNeurons = self.inputs*2
        #self.hiddenNeurons = self.inputs
        self.outputNeurons = outputs
        
        self.weightsOne = np.zeros((self.hiddenNeurons, self.inputs), dtype=float)
        self.weightsTwo = np.zeros((self.outputNeurons, self.hiddenNeurons), dtype=float)


        self.biasesOne = np.zeros(self.hiddenNeurons, dtype=float)
        self.biasesTwo = np.zeros(self.outputNeurons, dtype=float)
        
        #calculates the size of the necessary algorithm solutions to assign all weights and biases
        self.solutionSize = ((self.inputs*self.hiddenNeurons)) + self.hiddenNeurons + (self.hiddenNeurons * self.outputNeurons) + self.outputNeurons

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
    
    def forwardPass(self, inputs):
        #forward pass - preactivation(preA) and activation(H) - standard NN notation
        #create arrays to hold upcoming preactivation totals acting on the neurons
        preA = np.zeros((self.hiddenNeurons, self.inputs))
        
        #each output neuron has x number of inputs acting on it
        preA2 = np.zeros((self.outputNeurons, self.hiddenNeurons))
        
        #array to hold hidden neuron activation totals
        H = np.zeros(self.hiddenNeurons)
        
        #array for final outputs from output neurons
        finalOutputs = np.zeros(self.outputNeurons, dtype=float)
        
        #calculate the output of each neuron in the hidden layer
        for n in range(self.hiddenNeurons):
            for j in range(self.inputs):
                preA[n][j] = inputs[j] * self.weightsOne[n][j]
    
            hOut = self.sigmoid(np.sum(preA[n]) + self.biasesOne[n])
            H[n] = hOut
        
        #calculate the output of each neuron in the output layer
        for n in range(self.outputNeurons):
            for j in range(len(H)):
                preA2 [n][j] = H[j] * self.weightsTwo[n][j]

            finalOutputs[n] = self.sigmoid(np.sum(preA2[n]) + self.biasesTwo[n])
            
        return finalOutputs

    #decode an individual passed from an evolutionary algorithm
    def decodeEA(self, individual):
        #overall index keeps track of decoding progress
        index = 0
        
        #decode the hidden neuron weights and biases
        for i in range(self.hiddenNeurons):
            for j in range(self.inputs):
                self.weightsOne[i][j] = individual[index]
                index +=1
            self.biasesOne[i] = individual[index]
            index += 1
        
        #decode the weights and biases for the output neurons
        for k in range(self.outputNeurons):
            for l in range(self.hiddenNeurons):
                self.weightsTwo[k][l] = individual[index]
                index += 1
                
            self.biasesTwo[k] = individual[index]
            index += 1