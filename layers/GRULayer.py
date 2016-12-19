from layers.Layer import Layer
from activations import Linear, Sigmoid
import numpy as np
class GRULayer:
    def __init__(self,hidden_size,input_size,activation=Linear):
        self.hidden_size = hidden_size
        self.layer = Layer(hidden_size,[hidden_size,input_size],activation=activation)
        self.update = Layer(hidden_size,[hidden_size,input_size],activation=Sigmoid)
        self.reset = Layer(hidden_size,[hidden_size,input_size],activation=Sigmoid)
        self.stack = []

    def forward(self,inputs, hprev=None,train=False):
        if hprev is None:
            hprev = np.zeros((self.hidden_size,1))
        acts = []
        for input in inputs:
            z = self.update.forward([hprev,input],train=train)
            r = self.reset.forward([hprev,input],train=train)
            new_mem = self.layer.forward([r*(hprev),input],train=train)
            act = (1-z)*(new_mem) + z*(hprev)
            if train:
                self.stack.append((z,r,new_mem,act,hprev))
            hprev = act
            acts.append(act)
        return acts

    def backward(self,errors, alpha=0.01):
        dhnext = np.zeros((self.hidden_size,1))
        errors_out = []
        for error in errors[::-1]:
            error += dhnext
            (z,r,new_mem,act,hprev) = self.stack.pop()
            d_update, d_reset, dhnext, err_lower = 0, 0, 0, 0
            d_update += (hprev-new_mem)*error
            dhnext += z*error
            d_layer = error*(1-z)
            temp_1, temp_2 = self.layer.backward(d_layer,apply=False)
            err_lower += temp_2
            d_reset += temp_1*hprev
            dhnext += temp_1*r
            temp_1, temp_2 = self.reset.backward(d_reset,apply=False)
            dhnext += temp_1
            err_lower += temp_2
            temp_1,temp_2 = self.update.backward(d_update,apply=False)
            dhnext += temp_1
            err_lower += temp_2
            errors_out.append(err_lower)
        return errors_out[::-1]

    def apply_gradients(self,alpha = 0.01):
        self.layer.apply_gradients(alpha)
        self.update.apply_gradients(alpha)
        self.reset.apply_gradients(alpha)

if __name__ == "__main__":
    rnnlayer = GRULayer(hidden_size=3,input_size=2)
    inputs = [[0,1],[1,0],[1,1],[0,0]]
    print rnnlayer.forward(inputs)
