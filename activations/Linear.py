from __future__ import division
import numpy as np

def apply(input_):
    return np.maximum(input_,0)

def differentiate(activation):
    return np.ones(max(activation.shape)).reshape(activation.shape)*(activation>0)