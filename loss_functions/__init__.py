import CrossEntropy,MeanSquared

class Loss:
    def __init__(self,_type):
        self.compute = _type.compute
        self.error = _type.error

CrossEntropy = Loss(CrossEntropy)
MeanSquared = Loss(MeanSquared)
