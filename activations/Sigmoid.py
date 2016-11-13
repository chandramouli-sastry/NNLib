from __future__ import division
import numpy as np
def apply(input_):
    return 1 / (1 + np.exp(-1 * input_))

def differentiate(activation):
    return activation * (1 - activation)
