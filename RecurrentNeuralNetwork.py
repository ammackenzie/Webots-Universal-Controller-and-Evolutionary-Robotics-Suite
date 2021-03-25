import numpy as np

'''a basic implementation of an Elman Neural Network - a RNN that uses the output of the hidden layer as inputs for the next iteration
Network is designed with one layer of hidden neurons, set to be twice the number of input neurons
The context neurons are set to same number of hidden neurons
'''


class RecurrentNeuralNetwork():
    def __init__(self, inputs, outputs = 2):
        self.inputs = inputs
        self.hidden_neurons = self.inputs*2
        #self.hidden_neurons = self.inputs
        self.context_neurons = self.hidden_neurons
        self.output_neurons = outputs

        self.total_inputs = np.zeros((self.context_neurons + self.inputs), dtype=float)
        self.weight_one = np.zeros((self.hidden_neurons, len(self.total_inputs)), dtype=float)
        self.weights_two = np.zeros((self.output_neurons, self.hidden_neurons), dtype=float)

        self.context_inputs = np.zeros((self.hidden_neurons), dtype=float)

        self.biases_one = np.zeros(self.hidden_neurons, dtype=float)
        self.biases_two = np.zeros(self.output_neurons, dtype=float)
        
        #calculates the size of the necessary algorithm solutions to assign all weights and biases
        self.solution_size = ((self.inputs + self.context_neurons)*(self.hidden_neurons)) + self.hidden_neurons + (self.hidden_neurons * self.output_neurons) + self.output_neurons

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
    
    def forwardPass(self, inputs):
        #forward pass - preactivation(preA) and activation(H) - standard NN notation
        
        #concatenate context neurons saved inputs from last iteration and current new inputs
        self.total_inputs = np.concatenate((inputs, self.context_inputs))
        
        #Rectifies an issue with Webots sometimes reading 0.0 as nan
        np.nan_to_num(self.total_inputs, False)
        
        #create arrays to hold upcoming preactivation totals acting on the neurons
        preA = np.zeros((self.hidden_neurons, len(self.total_inputs)))
        
        #each output neuron has x number of inputs acting on it
        preA2 = np.zeros((self.output_neurons, self.hidden_neurons))
        
        #array to hold hidden neuron activation totals
        H = np.zeros(self.hidden_neurons)
        
        #array for final outputs from output neurons
        final_outputs = np.zeros(self.output_neurons, dtype=float)
        
        for neuron in range(self.hidden_neurons):
            for input in range(len(self.total_inputs)):
                preA[neuron][input] = self.total_inputs[input] * self.weight_one[neuron][input]
    
            hOut = self.sigmoid(np.sum(preA[neuron]) + self.biases_one[neuron])
            H[neuron] = hOut
            #save hidden layer output as context inputs for next iteration
            self.context_inputs[neuron] = hOut

        for neuron in range(self.output_neurons):
            for input in range(len(H)):
                preA2 [neuron][input] = H[input] * self.weights_two[neuron][input]

            final_outputs[neuron] = self.sigmoid(np.sum(preA2[neuron]) + self.biases_two[neuron])
            
        return final_outputs

    #decode an individual passed from an evolutionary algorithm
    def decodeEA(self, individual):
        index = 0

        for i in range(self.hidden_neurons):
            for input in range(len(self.total_inputs)):
                self.weight_one[i][input] = individual[index]
                index +=1
            self.biases_one[i] = individual[index]
            index += 1

        for k in range(self.output_neurons):
            for l in range(self.hidden_neurons):
                self.weights_two[k][l] = individual[index]
                index += 1
                
            self.biases_two[k] = individual[index]
            index += 1