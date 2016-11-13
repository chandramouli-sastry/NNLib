from layers.Layer import Layer
from layers.SoftmaxLayer import SoftmaxLayer
import numpy as np
from loss_functions import CrossEntropy

class ANN:
    def __init__(self,list_sizes,input_size, loss_fn=CrossEntropy):
        prev_size = input_size
        self.alpha = 1.0
        self.layers = []
        for i in list_sizes[:-1]:
            self.layers.append(Layer(i,[prev_size]))
            prev_size = i
        self.layers.append(SoftmaxLayer(list_sizes[-1],[prev_size]))
        self.loss_fn = loss_fn

    def forward(self,input_,train= False):
        inp = input_
        for layer in self.layers:
            inp = layer.forward([inp], train)
        return inp

    def backward(self,observed, expected, epoch):
        if epoch%5==0:
            self.alpha -= 0.001
        expected = np.array(expected).reshape((len(expected),1))
        error = self.loss_fn.error(observed, expected)
        #error = self.layers[-1].backward(error,np.argmax(expected))[0]
        for layer in reversed(self.layers):
            error = layer.backward(error, alpha = self.alpha)[0]

examples = [[[0,1],[0,1]],[[1,0],[1,0]]]#,[[0,0],[1,0]],[[1,1],[1,0]]]
ann = ANN([6,2],2)
for epoch in range(100):
    for example,target in examples:
        observed = ann.forward(example,train = True)
        ann.backward(observed,target,epoch)
for example,target in examples:
    print example, target, ann.forward(example)

