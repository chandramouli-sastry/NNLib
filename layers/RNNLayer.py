from layers.Layer import Layer
from activations import Linear
import numpy as np
class RNNLayer:
    def __init__(self,hidden_size,input_size,activation=Linear):
        self.hidden_size = hidden_size
        self.layer = Layer(hidden_size,[hidden_size,input_size],activation=activation)

    def forward(self,inputs, dhprev=None,train=False):
        if dhprev is None:
            dhprev = np.zeros(self.hidden_size)
        acts = []
        for input in inputs:
            act = self.layer.forward([dhprev,input],train=train)
            dhprev = act
            acts.append(act)
        return acts

    def backward(self,errors, alpha=0.01):
        dhnext = np.zeros((self.hidden_size,1))
        errors_out = []
        for error in errors[::-1]:
            dhnext,err = self.layer.backward(error+dhnext,alpha=alpha,apply=False)
            errors_out.append(err)
        return errors_out[::-1]

    def apply_gradients(self,alpha = 0.01):
        self.layer.apply_gradients(alpha)

if __name__ == "__main__":
    rnnlayer = RNNLayer(hidden_size=3,input_size=2)
    inputs = [[0,1],[1,0],[1,1],[0,0]]
    print rnnlayer.forward(inputs)
