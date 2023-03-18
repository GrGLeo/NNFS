import numpy as np

#ReLU activation
class Activation_ReLU:

    #Forward pass
    def forward(self,inputs):
        self.inputs = inputs
        #calculate output values from input
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0



# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        # print(inputs)
        # print(np.max(inputs, axis=1,keepdims=True))
        exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities

    # Backdward pass
    def backward(self,dvalues):
        #Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        #Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #Flatten output array
            single_output = single_output.reshape(-1, 1)
            #Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #Calculate sample_wise gradient
            #and add it to the array of the sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Activation_Sigmoid:

    # Forward pass
    def forward(self,inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self,dvalues):
        self.dinputs = dvalues *  (1 - self.output) * self.output
