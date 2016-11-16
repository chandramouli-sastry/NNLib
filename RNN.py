from __future__ import division
import numpy as np
from activations import Linear
from layers.RNNLayer import RNNLayer
from layers.Layer import Layer
from loss_functions import MeanSquared
np.random.seed(1)
class RNN:
    def __init__(self,layers, input_size, cost_function):
        self.alpha = 1.0
        self.layers = []
        self.input_size = input_size
        prev = input_size
        for Layer,i,act in layers[:-1]:
            l = Layer(hidden_size=i,input_size=prev,activation=act)
            prev = i
            self.layers.append(l)
        Layer,i,act = layers[-1]
        self.layers.append(Layer(i,[prev],activation=act))
        self.cost_function = cost_function
        for ind,i in enumerate(self.layers):
            print ind
            try:
                print "Weights"
                for j in i.weights:
                    print j
                print "bias"
                print i.bias
            except Exception as e:
                i=i.layer
                print "Weights"
                for j in i.weights:
                    print j
                print "bias"
                print i.bias
        #input("aaaa")

    def forward(self,inputs,train=False):
        for layer in self.layers[:-1]:
            inputs = layer.forward(inputs,train=train)
        return [self.layers[-1].forward([input],train=train) for input in inputs]


    def backward(self,observed, expected, epoch, max_epoch):
        self.alpha = self.alpha - epoch/max_epoch
        errors = [0]*len(observed)
        outer_layer = self.layers[-1]
        cost = 0
        for ind in range(len(observed)-1,-1,-1):
            err = self.cost_function.error(observed[ind],expected[ind])
            cost += self.cost_function.compute(observed[ind],expected[ind])
            errors[ind]=outer_layer.backward(err,apply=False)[0]
        for layer in self.layers[:-1][::-1]:
            errors = layer.backward(errors)
        for layer in self.layers:
            layer.apply_gradients(self.alpha)
     #   print "1"
        return cost

if __name__=="__main__":
    length = 3
    rnn = RNN(layers=[(RNNLayer,3,Linear),(Layer,1,Linear)],input_size=1,cost_function=MeanSquared)
    samples = [(map(lambda x:[int(x)],'0'*(length-len(bin(i))+2)+bin(i)[2:])) for i in range(0,2**length)]
    for ind,l in enumerate(samples):
        s = l[0]
        l1=[s[0]]
        for t in l:
            l1.append(2*l1[-1]+t[0])
        samples[ind]=l,map(lambda x:[x],l1[1:])
    samples=[samples[2]]
    print samples
    max_epoch = 1000
    for epoch in range(max_epoch):
        cost = 0
        for observation,target in samples:
            x = rnn.forward(observation,train=True)
            cost += rnn.backward(x,target,epoch,max_epoch)
        if epoch%10==0:
            print "Epoch : ",epoch,"Cost: ",cost
    while True:
        x = raw_input("Enter sequence")
        x = map(lambda x:[int(x)],x)
        print rnn.forward(x,train=False)

