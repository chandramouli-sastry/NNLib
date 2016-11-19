import Sigmoid,Linear,Softmax

class Act:

    def __init__(self,_type):
        self.apply=_type.apply
        self.differentiate = _type.differentiate

Sigmoid = Act(Sigmoid)
Linear = Act(Linear)
Softmax = Act(Softmax)