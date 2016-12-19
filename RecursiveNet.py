from layers.Layer import Layer
from activations import Linear
embeddings = []
class RecursiveNet:
    def __init__(self,d):
        self.layer = [Layer(layer_size=1,input_sizes=[d,d],activation=Linear),Layer(layer_size=d,input_sizes=[d,d])]

    def forward(self,vector1, vector2):
        score = self.layer[0].forward([vector1,vector2])
        output_vector = self.layer[1].forward([vector1,vector2])

    def backward(self,error,base=False):
        if not base:
            self.layer[0].backward(error,apply=True)
            self.layer[1].backward(error,apply=True)
        else:
            #update embeddings
            pass
