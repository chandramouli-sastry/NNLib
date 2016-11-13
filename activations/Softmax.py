from __future__ import division
import numpy as np
def apply(input_):
    return np.exp(input_)/np.exp(input_).sum()

def differentiate(activation,target):
    shape = activation.shape
    activation = activation.reshape(1,len(activation)).tolist()[0]
    for i in xrange(len(activation)):
        if i!=target:
            activation[i] *= -activation[target]
        else:
            activation[i] *= (1-activation[target])

    return np.array(activation).reshape(shape)
