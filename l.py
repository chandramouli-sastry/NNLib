
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
            if True or epoch%5==0:
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
    test(_type)
    examples,ann = get_ann("Auto")
    max_epochs = 5000
    for epoch in range(max_epochs):
        for example,target in examples:
            observed = ann.forward(example,train=True)
            ann.backward(observed,target,0,max_epochs,apply=True)
            #ann.apply_grads()

    for example,target in examples:
         observed = ann.forward(example)
         print(example,target, observed)

    #test(_type)