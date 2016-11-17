from __future__ import division
import numpy as np

#Must be numpy
def compute(observed,target):

    return 0.5 * ((observed-target)**2).sum()

def error(observed,target):
    return (target-observed)

