import numpy as np

# class Activation_Function:
#     def __init__(self):
# 
#         self.inputs = None
#         self.output = None
# 
#     def forward(self,inputs):
#         self.inputs = inputs
#         self.output = function(inputs)

class Activation_ReLU:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        dinputs = dvalues.copy()
        dinputs[self.inputs <= 0] = 0

        return dinputs

class Activation_Linear:
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
        return self.output
    
    def backward(self, dvalues):
        return dvalues