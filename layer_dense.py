#Dense layer
import numpy as np

class Layer_Dense:
    #layer initialization
    def __init__(self,n_inputs,n_neurons,
                 weight_regularizer_l1=0,weight_regularizer_l2=0,
                 bias_regularizer_l1=0,bias_regularizer_l2=0):
        #Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        #Set regularizer strenght
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    #Forward pass
    def forward(self,inputs):
        #Calculate the output value from inputs, weights, and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        # print(self.weights)

    #Backward pass
    def backward(self,dvalues):
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0,keepdims=True)
        #Gradients on values
        self.dinputs = np.dot(dvalues,self.weights.T)
