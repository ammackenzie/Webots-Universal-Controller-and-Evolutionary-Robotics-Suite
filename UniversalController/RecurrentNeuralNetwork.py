import numpy as np

'''a basic implementation of an Elman Neural Network - a RNN that uses the output of the hidden layer as inputs for the next iteration
Network is designed with one layer of hidden neurons, set to be twice the number of input neurons
The context neurons are set to same number of hidden neurons
'''


class RecurrentNeuralNetwork():
    def __init__(self, inputs, outputs = 2):
        self.inputs = inputs
        self.hiddenNeurons = self.inputs*2
        #self.hiddenNeurons = self.inputs
        self.contextNeurons = self.hiddenNeurons
        self.outputNeurons = outputs

        self.totalInputs = np.zeros((self.contextNeurons + self.inputs), dtype=float)
        self.weightsOne = np.zeros((self.hiddenNeurons, len(self.totalInputs)), dtype=float)
        self.weightsTwo = np.zeros((self.outputNeurons, self.hiddenNeurons), dtype=float)

        self.contextInputs = np.zeros((self.hiddenNeurons), dtype=float)

        self.biasesOne = np.zeros(self.hiddenNeurons, dtype=float)
        self.biasesTwo = np.zeros(self.outputNeurons, dtype=float)
        
        #calculates the size of the necessary algorithm solutions to assign all weights and biases
        self.solutionSize = ((self.inputs + self.contextNeurons)*(self.hiddenNeurons)) + self.hiddenNeurons + (self.hiddenNeurons * self.outputNeurons) + self.outputNeurons

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
    
    def forwardPass(self, inputs):
        #forward pass - preactivation(preA) and activation(H) - standard NN notation
        
        #concatenate context neurons saved inputs from last iteration and current new inputs
        self.totalInputs = np.concatenate((inputs, self.contextInputs))
        
        #Rectifies an issue with Webots sometimes reading 0.0 as nan
        np.nan_to_num(self.totalInputs, False)
        
        #create arrays to hold upcoming preactivation totals acting on the neurons
        preA = np.zeros((self.hiddenNeurons, len(self.totalInputs)))
        
        #each output neuron has x number of inputs acting on it
        preA2 = np.zeros((self.outputNeurons, self.hiddenNeurons))
        
        #array to hold hidden neuron activation totals
        H = np.zeros(self.hiddenNeurons)
        
        #array for final outputs from output neurons
        finalOutputs = np.zeros(self.outputNeurons, dtype=float)
        
        for n in range(self.hiddenNeurons):
            for j in range(len(self.totalInputs)):
                preA[n][j] = self.totalInputs[j] * self.weightsOne[n][j]
    
            hOut = self.sigmoid(np.sum(preA[n]) + self.biasesOne[n])
            H[n] = hOut
            #save hidden layer output as context inputs for next iteration
            self.contextInputs[n] = hOut

        for n in range(self.outputNeurons):
            for j in range(len(H)):
                preA2 [n][j] = H[j] * self.weightsTwo[n][j]

            finalOutputs[n] = self.sigmoid(np.sum(preA2[n]) + self.biasesTwo[n])
            
        return finalOutputs

    #decode an individual passed from an evolutionary algorithm
    def decodeEA(self, individual):
        index = 0

        for i in range(self.hiddenNeurons):
            for j in range(len(self.totalInputs)):
                self.weightsOne[i][j] = individual[index]
                index +=1
            self.biasesOne[i] = individual[index]
            index += 1

        for k in range(self.outputNeurons):
            for l in range(self.hiddenNeurons):
                self.weightsTwo[k][l] = individual[index]
                index += 1
                
            self.biasesTwo[k] = individual[index]
            index += 1