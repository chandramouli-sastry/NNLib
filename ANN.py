from __future__ import division
import pickle
from activations import Sigmoid
from layers.Layer import Layer
from activations import Linear
from layers.SoftmaxLayer import SoftmaxLayer
import numpy as np
from loss_functions import CrossEntropy,MeanSquared

class ANN:
    def __init__(self,list_sizes,input_size, loss_fn=CrossEntropy):
        prev_size = input_size
        self.alpha = 0.01
        self.layers = []
        for Layer,i,act in list_sizes:
            self.layers.append(Layer(i,[prev_size],activation = act))
            prev_size = i
        #self.layers.append(SoftmaxLayer(list_sizes[-1][1],[prev_size]))
        self.loss_fn = loss_fn

    def forward(self,input_,train= False):
        inp = input_
        for layer in self.layers:
            inp = layer.forward([inp], train)
        return inp

    def backward(self,observed, expected, epoch, max_epochs):
        if epoch%5==0:
            self.alpha -= epoch/max_epochs
        expected = np.array(expected).reshape((len(expected),1))
        error = self.loss_fn.error(observed, expected)
        #print observed,expected
        cost = self.loss_fn.compute(observed, expected)
        #print cost
        #raw_input()
        #error = self.layers[-1].backward(error,np.argmax(expected))[0]
        for layer in reversed(self.layers):
            error = layer.backward(error, alpha = self.alpha,apply=True)[0]
        return cost
def accuracy(samples,targets):
    count = 0
    for sample, target in zip(samples,targets):
       # print ann.forward(sample)
        #print target,ann.forward(sample)
        #raw_input()
        count += 1 if target==np.argmax(ann.forward(sample)) else 0
        #print count
        #raw_input()
    print count,len(samples)
train, validate, test = pickle.load(open("./mnist.pkl","rb"))
print len(train[0]),len(train[1])
ann = ANN([(Layer,25,Sigmoid),(Layer,10,Linear)],784,loss_fn = MeanSquared)
max_epochs = 1000
accuracy(train[0],train[1])
for epoch in range(max_epochs):
    cost = 0
    for index in range(len(train)):
        sample = train[0][index]
        target = train[1][index]
        expected = [0]*10
        expected[target] = 1
        observed = ann.forward(sample,train=True)
        cost += ann.backward(observed,expected,0,max_epochs)
    if epoch%5:
        print "Epoch: ",epoch, "Cost: ",cost

accuracy(train[0],train[1])
accuracy(test[0],test[1])
'''examples = [[[0,1],[0,1]],[[1,0],[1,0]]]#,[[0,0],[1,0]],[[1,1],[1,0]]]
ann = ANN([6,2],10)
for epoch in range(100):
    for example,target in examples:
        observed = ann.forward(example,train = True)
        ann.backward(observed,target,epoch)
for example,target in examples:
    print example, target, ann.forward(example)
'''
