from __future__ import division
import math
import numpy as np
from activations import Sigmoid,Linear
def fn(a,b):
    return np.random.uniform(low=-0.01,high=0.01,size=(a,b))
def xavier(a,b):
    return real_randn(a,b)*2.0/(math.sqrt(a)+math.sqrt(b))
real_randn = np.random.randn
np.random.randn = fn
class Layer:
    def __init__(self, layer_size, input_sizes, init_function = np.random.randn, activation=Sigmoid, alpha=0.01):
        self.alpha = alpha
        self.layer_size = layer_size
        self.input_sizes = input_sizes
        self.weights = [np.random.randn(layer_size, input_size)#*2.0/(math.sqrt(input_size)+math.sqrt(layer_size)) \
                         for input_size in input_sizes]
        self.mems = [np.zeros_like(weight) for weight in self.weights]
        self.deltas = [np.zeros((layer_size,input_size)) for input_size in input_sizes]
        if activation==Linear:
            self.bias =  np.ones((layer_size,1))
        else:
            self.bias = fn(layer_size, 1)
        self.del_bias = np.zeros((layer_size,1))
        self.mem_bias = np.zeros_like(self.del_bias)
        self.input_stack = []
        self.activation_stack = []
        self.activation = activation

    def print_w(self):
        print "Init weight : ",self.weights
        print "Init bias : ",self.bias

    def forward(self, input_list,train = False):
        temp = self.bias
        for index, input_ in enumerate(input_list):
            if type(input_)==type([]):
                input_ = np.array(input_)
            reshaped = input_.reshape((len(input_),1))
            input_list[index] = reshaped
            temp = temp + self.weights[index].dot(reshaped)
        act = self.activation.apply(temp)
        if train:
            self.activation_stack.append(act)
            self.input_stack.append(input_list) #WARNING: input list is pushed without copying
        #print "AC",self.activation_stack,len(self.activation_stack),"/AC"
        #print "IN",self.input_stack,len(self.input_stack),"/IN"
        return act

    def backward(self, error_incoming, apply= True, alpha = 0.01):
        error_incoming = np.array(error_incoming).reshape((self.layer_size,1)) if type(error_incoming)==type([]) else error_incoming
        error_lower = error_incoming * self.activation.differentiate(self.activation_stack.pop())
        inputs = self.input_stack.pop()
        self.compute_gradients(error_lower, inputs)
        error_outgoing = [weight.T.dot(error_lower) for weight in self.weights]#weight: inp*layer_size layer_size*1
        if apply:
            self.apply_gradients(alpha)
        return error_outgoing

    def compute_gradients(self, error_lower, inputs):
        for index, input_ in enumerate(inputs):
            self.deltas[index] = self.deltas[index] + error_lower.dot(input_.T)
        self.del_bias = self.del_bias + error_lower

    def apply_gradients(self, alpha = 0.01):
        for index,delta in enumerate(self.deltas):
            self.mems[index] += delta*delta
            self.weights[index] += alpha * delta.clip(-5,5)/np.sqrt(self.mems[index]+1e-8)
        self.mem_bias += self.del_bias*self.del_bias
        self.bias += alpha * self.del_bias/np.sqrt(self.mem_bias+1e-8)
        self.deltas = [np.zeros((self.layer_size,input_size)) for input_size in self.input_sizes]
        self.del_bias = np.zeros((self.layer_size,1))


def test(_type):
    max_max_epochs = 10000
    for h in range(max_max_epochs):
        examples, ann = get_ann(_type)
        max_epochs = 10
        for epoch in range(max_epochs):
            for example, target in examples:
                observed = ann.forward(example, train=True)
                ann.backward(observed, target, epoch, max_epochs)
        max = -1
        for example, target in examples:
            forward = ann.forward(example)
            print max,forward
            print forward>max
            if (forward > max):
                max = forward

        if (max == 0):
            print("FAIL", h)
            for example, target in examples:
                print(ann.forward(example))
            ann.print_w()
            break
        if (h % 1000 == 0):
            print(h)

if __name__=="__main__":
    '''x = Layer(2,[1],activation=Linear)
    for e in range(1000):
        p = x.forward([[1]],train=True)
        #print(p)
        err = np.array([[3],[2]])-p
        x.backward(err)
    print x.forward([[1]])
    input("Satisfied?")'''
    log = False#True
    def get_ann(_type):
        if _type=="Auto":
            examples = [[[0,1],[0,1]],[[0,0],[0,0]],[[1,0],[1,0]],[[1,1],[1,1]]]
            ann = ANN([(Layer,5,Linear),(Layer,2,Linear)],2)
        elif _type=="Bin-To-Decimal":
            examples = [[[0,1],[1]],[[0,0],[0]],[[1,0],[2]],[[1,1],[3]]]
            ann = ANN([(Layer,4,Linear),(Layer,1,Linear)],2)
        elif _type=="Bin2":
            examples = [[[0,1],[1]],[[0,0],[0]],[[1,0],[2]],[[1,1],[3]]]
            ann = ANN([(Layer,4,Linear),(Layer,1,Linear)],2)
        else:
            examples = [[[0,1],[1]],[[0,0],[0]],[[1,0],[1]],[[1,1],[0]]]
            ann = ANN([(Layer,3,Linear),(Layer,1,Linear)],2)
        return examples,ann
    class ANN:
        def __init__(self,list_sizes,input_size):
            self.alpha = 0.01
            prev_size = input_size
            self.layers = []
            for Layer,i,act in list_sizes:
                self.layers.append(Layer(i,[prev_size],activation=act))
                prev_size = i

        def forward(self,input_,train=False):
            inp = input_
            for layer in self.layers:
                inp = layer.forward([inp],train=train)
            if log:
                print "fwd"
                for layer in self.layers:
                    print "---"
                    print str(layer.activation_stack),str(layer.input_stack)
                    print "---"
            return inp

        def backward(self,observed, expected, epoch, max_epochs,apply=True):
            if False and epoch%5==0:
                self.alpha -= epoch/max_epochs
            expected = np.array(expected).reshape((len(expected),1))
            error = ( expected - observed )
            for layer in reversed(self.layers):
#                print apply
                error = layer.backward(error,alpha=self.alpha,apply=apply)[0]
            if log:
                print "bwd"
                for layer in self.layers:
                    print "---"
                    print str(layer.activation_stack),str(layer.input_stack)
                    print "---"
        def apply_grads(self):
            for i in self.layers:
                i.apply_gradients(self.alpha)

        def print_w(self):
            for layer in self.layers:
                layer.print_w()

    _type = "Auto"
    #test(_type)
    examples,ann = get_ann("Auto")
    max_epochs = 5000
    for epoch in range(max_epochs):
        for example,target in examples:
            observed = ann.forward(example,train=True)
            ann.backward(observed,target,0,max_epochs,apply=False)
            ann.apply_grads()

    for example,target in examples:
         observed = ann.forward(example)
         print(example,target, observed)

    #test(_type)