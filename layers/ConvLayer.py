from __future__ import division
from activations import Sigmoid,Tanh
from layers.ConvFilter import ConvFilter
from layers.Layer import Layer
from activations import Linear
from layers.SoftmaxLayer import SoftmaxLayer
import numpy as np
from loss_functions import CrossEntropy,MeanSquared

class ConvLayer:
    def __init__(self,filters,input_size,activation=Tanh):
        self.filters = []
        self.input_size = input_size
        self.activation = activation
        for window,stride in filters:
            fil = ConvFilter(input_size = input_size, window_size = window, stride=stride,activation=activation,pooling=True)
            self.filters.append(fil)

    def forward(self, input_list, train=False):
        acts = []
        for filter in self.filters:
            act = filter.forward(input_list,train)
            acts.append(act)
        return acts

    def backward(self,errors,alpha = 0.01,apply=False):
        errors = errors.tolist()
        for index,error in enumerate(errors):
            self.filters[index].backward(np.array([error[0]]).reshape((1,1)),apply,alpha=alpha)

    def apply(self,alpha = 0.01):
        for filter in self.filters:
            filter.apply(alpha)
