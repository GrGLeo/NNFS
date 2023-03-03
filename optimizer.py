import numpy as np
#SGD Optimizer
class Optimizer_SGD:
    #Initialize optimizer - set settings
    # learning rate of 1 is the default parameter
    def __init__(self, learning_rate= 1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.momentum = momentum

    #Call once before parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay*self.iteration))

    #Update parameter
    def update_params(self,layer):
        #If we use momentum
        if self.momentum:
            if not hasattr(layer,'weights_momentum'):
                layer.weights_momentum = np.zeros_like(layer.weights)
                layer.biases_momentum = np.zeros_like(layer.biases)
            weights_update = self.momentum * layer.weights_momentum - self.current_learning_rate * layer.dweights
            layer.weights_momentum = weights_update
            biases_update = self.momentum * layer.biases_momentum - self.current_learning_rate * layer.dbiases
            layer.biases_momentum = biases_update
        #If we do not use momentum
        else:
            weights_update = -self.current_learning_rate * layer.dweights
            biases_update = -self.current_learning_rate * layer.dbiases

        #weihgts and biases update
        layer.weights += weights_update
        layer.biases += biases_update
    
    #Call once after parameters update
    def post_update_params(self):
        self.iteration+=1