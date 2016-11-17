import numpy as np
from activations import Softmax
from layers.Layer import Layer


class SoftmaxLayer(Layer):
    def __init__(self, layer_size, input_sizes, init_function = np.random.randn, alpha=0.1,activation=None):
        Layer.__init__(self,layer_size, input_sizes,
                       init_function = np.random.randn,activation=Softmax, alpha=0.1)


    def backward(self, error_incoming, apply= True, alpha = 0.01):
        error_incoming = np.array(error_incoming).reshape((self.layer_size,1)) if type(error_incoming)==type([]) else error_incoming
        act = self.activation_stack.pop()
        error_lower = np.array([(error_incoming * self.activation.differentiate(act,i)).sum() for i in xrange(self.layer_size)]).reshape((self.layer_size,1))
        #print error_lower
        #raw_input()
        inputs = self.input_stack.pop()
        self.compute_gradients(error_lower, inputs)
        error_outgoing = [weight.T.dot(error_lower) for weight in self.weights]
        if apply:
            self.apply_gradients(alpha)
        return error_outgoing

