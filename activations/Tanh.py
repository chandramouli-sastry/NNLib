from __future__ import division
import numpy as np
def apply(input_):
    return np.tanh(input_)

def differentiate(activation):
    return (1 - activation*activation)

