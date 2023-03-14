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


#AdaGrad Optimizer
class Optimizer_AdaGrad:
    #Initialize optimizer - set settings
    # learning rate of 1 is the default parameter
    def __init__(self, learning_rate= 1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon

    #Call once before parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay*self.iteration))

    #Update parameter
    def update_params(self,layer):
        #If we use momentum
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache +=layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        #weihgts and biases update
        layer.weights += -self.current_learning_rate*layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    #Call once after parameters update
    def post_update_params(self):
        self.iteration+=1


#RMSprop Optimizer
class Optimizer_RMSprop:
    #Initialize optimizer - set settings
    # learning rate of 1 is the default parameter
    def __init__(self, learning_rate= 0.001, decay=0., epsilon=1e-7,rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon
        self.rho = rho

    #Call once before parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay*self.iteration))

    #Update parameter
    def update_params(self,layer):
        #If we use momentum
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho*layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho*layer.bias_cache + (1-self.rho) * layer.dbiases**2

        #weihgts and biases update
        layer.weights += -self.current_learning_rate*layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    #Call once after parameters update
    def post_update_params(self):
        self.iteration+=1


#Adam Optimize
class Optimizer_Adam:
    #Initialize optimizer - set settings
    # learning rate of 1 is the default parameter
    def __init__(self, learning_rate= 0.001, decay=0., epsilon=1e-7,beta1=0.9,beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    #Call once before parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay*self.iteration))

    #Update parameter
    def update_params(self,layer):
        #If we use momentum
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with currents gradients
        layer.weight_momentum = self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.dweights
        layer.bias_momentum = self.beta1 * layer.bias_momentum + (1 - self.beta1) * layer.dbiases
        # Get corrected momentum
        weight_momentum_corrected = layer.weight_momentum / (1 - self.beta1 ** (self.iteration + 1))
        bias_momentum_corrected = layer.bias_momentum / (1 - self.beta1 ** (self.iteration +1))

        # Update cache with currents grandients
        layer.weight_cache = self.beta2*layer.weight_cache + (1-self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2*layer.bias_cache + (1-self.beta2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iteration + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iteration + 1))

        #weihgts and biases update
        layer.weights += -self.current_learning_rate*weight_momentum_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate*bias_momentum_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)

    #Call once after parameters update
    def post_update_params(self):
        self.iteration+=1
