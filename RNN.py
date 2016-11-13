import numpy as np
class RNN:
    def __init__(self,layers, input_size):
        self.layers = []
        self.input_size = input_size
        prev = input_size
        for Layer,i,act in layers:
            l = Layer([i,prev],activation=act)
            prev = i
            self.layers.append(l)

    def forward(self,inputs):

        for layer in self.layers:
            dhprev = [0] * layer.layer_size
            acts = []
            for i in inputs:
                act = layer.forward([dhprev,i])
                dhprev = act
                acts.append(act)
            inputs = acts
        return acts


