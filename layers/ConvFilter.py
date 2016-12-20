from activations import Tanh
from layers.Layer import Layer
from activations import Linear
import numpy as np
class ConvFilter(Layer):
    def __init__(self,input_size,window_size,stride,activation=Tanh,pooling = False):
        self.window_size = window_size
        self.stride = stride
        Layer.__init__(self,1,[input_size]*window_size,activation=activation)
        self.pooling = pooling


    def forward(self, input_list,train = False):
        acts = []
        if(len(input_list) < self.window_size):
            return np.array([0]).reshape((1,1))
        for i in range(0,len(input_list),self.stride):
            input_ = input_list[i:i+self.window_size]
            act = Layer.forward(self,input_,train = train)[0][0]
            acts.append(act)
        if(self.pooling):
            acts  = np.array([max(acts)]).reshape((1,1))
            self.activation_stack = [acts]
        return acts



