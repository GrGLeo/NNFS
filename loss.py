import numpy as np

from activations import Activation_Softmax

#Common loss class
class Loss:
    #Calculate the data and regularization losses
    #given model output and ground thruth values
    def calculate(self,output,y):
        #calculate sample losses
        sample_losses = self.forward(output, y)

        #calculate mean loss
        data_loss = np.mean(sample_losses)

        #Return loss
        return data_loss

#Cross_entropy class

class Loss_CategoricalCrossentropy(Loss):

    #forward pass
    def forward(self, y_pred,y_true):
        #number of sample in a batch
        samples = len(y_pred)

        #Clip data to prevent division by 0
        #Clip both side to not drag mean toward any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #Probabilites for taget values
        #only categorical labels
        if len(y_true.shape) ==1:
            correct_confidences = y_pred_clipped[
                                            range(samples),
                                            y_true]

        #Mask values- only for one hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,
                                        axis=1)

        #Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    #backward pass
    def backward(self,dvalues,y_true):
        #Number of samples
        samples = len(dvalues)

        #Number of labels in every sample
        #We'll use the first sample to count them
        labels=len(dvalues[0])
        #If laberls are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        #Calculate gradient
        self.dinputs = -y_true / dvalues
        #Normalize gradient
        self.dinputs = self.dinputs / samples

#Softmax classifier -combined Softmax activation
# and cross_entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    #create activations and loss function object
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    #Forward pass
    def forward(self, inputs, y_true):
        #Output layer's activation function
        self.activation.forward(inputs)
        #Set the output
        self.output = self.activation.output
        #Calculate and return the loss
        return self.loss.calculate(self.output,y_true)

    # Backward pass
    def backward(self, dvalues, y_true):

        #Number of sample
        samples = len(dvalues)

        #If labels are one-hot encoded
        #Turn them into labels
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        #copy so we can safely modify
        self.dinputs = dvalues.copy()
        #calculate gradient
        self.dinputs[range(samples),y_true] -= 1
        #Normalize gradients
        self.dinputs = self.dinputs / samples
        
        