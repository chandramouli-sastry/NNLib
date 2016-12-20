from __future__ import division
import pickle
from activations import Sigmoid
from layers.ConvLayer import ConvLayer
from layers.Layer import Layer
from activations import Linear
from layers.SoftmaxLayer import SoftmaxLayer
import numpy as np
from loss_functions import CrossEntropy,MeanSquared

class CNN:
    def __init__(self,layers,input_size,cost_function=CrossEntropy):
        prev= input_size
        self.layers = []
        for layer,attrs in layers:
            if type(layer) == ConvLayer:
                l = ConvLayer(filters=attrs[0],input_size=prev,activation=attrs[1])
            else:
                l = layer(layer_size=attrs[0],input_size=prev,activation=attrs[1])
            prev = attrs[0]
            self.layers.append(l)
        self.cost_fn = cost_fn

    def forward(self,inputs,train=False):
        for layer in self.layers:
            inputs = layer.forward(inputs,train = train)
        return inputs

    def backward(self,observed,expected,epoch,max_epoch):
       error = -( observed - expected )
       cost = self.cost_fn.compute(observed,expected)
       for layer in self.layers[::-1]:
           error = layer.backward(error,apply = True)
       return cost,error



