from __future__ import division
import numpy as np
from activations import Linear, Sigmoid
from layers.RNNLayer import RNNLayer
from layers.Layer import Layer
from layers.SoftmaxLayer import SoftmaxLayer
from loss_functions import MeanSquared, CrossEntropy
#np.random.seed(1)
class RNN:
    def __init__(self,layers, input_size, cost_function):
        self.alpha = 0.1
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
        if False:
            self.alpha = self.alpha - epoch/max_epoch
        errors = [0]*len(observed)
        outer_layer = self.layers[-1]
        cost = 0
        """for ind in range(len(observed)-1,-1,-1):
            if type(expected[ind])==type([]):
                expected[ind]=np.array(expected[ind])
            expected[ind] = expected[ind].reshape(observed[ind].shape)
            #print expected[ind],observed[ind]
            err = self.cost_function.error(observed[ind],expected[ind])
            cost += self.cost_function.compute(observed[ind],expected[ind])
            errors[ind]=outer_layer.backward(err,apply=False)[0]"""
        flag = True
        for ind in range(len(observed)-1,-1,-1):
            if type(expected[ind])==type([]):
                expected[ind]=np.array(expected[ind])
            expected[ind] = expected[ind].reshape(observed[ind].shape)
            #if(not flag):
                #expected[ind] = observed[ind]
            cost+=self.cost_function.compute(observed[ind],expected[ind])
            err = -( observed[ind] - expected[ind] )
            errors[ind]=outer_layer.backward(err,apply=False)[0]
            flag =False

        for layer in self.layers[:-1][::-1]:
            errors = layer.backward(errors)
        for layer in self.layers:
            layer.apply_gradients(self.alpha)
     #   print "1"
        return cost
    #Let me disconnect and reconnect
if __name__=="__main__":
    length = 5
    rnn = RNN(layers=[(RNNLayer,3,Sigmoid),(SoftmaxLayer,2,None)],input_size=1,cost_function=CrossEntropy)
    samples = [(map(lambda x:[int(x)],'0'*(length-len(bin(i))+2)+bin(i)[2:])) for i in range(0,2**length)]
    for ind,sample in enumerate(samples):
        op = [sample[0]]
        for i in sample[1:]:
            op.append([op[-1][0]^i[0]])
        for ind1,i in enumerate(op):
            op[ind1]=[0,1] if i[0]==1 else [1,0]
        samples[ind]=(sample,op)
    max_epoch = 1000
    for epoch in range(max_epoch):
        cost = 0
        costs = []
        for observation,target in samples:
            x = rnn.forward(observation,train=True)
            backward = rnn.backward(x, target, epoch, max_epoch)
            cost += backward
            costs.append(backward)
        if epoch%1==0:
            print "Epoch : ",epoch,"Cost: ",cost
    while True:
        x = raw_input("Enter sequence")
        x = map(lambda x:[int(x)],x)
        v = rnn.forward(x,train=False)
        for ind in range(len(v)):
            v[ind] = np.argmax(v[ind])
        print(v)

