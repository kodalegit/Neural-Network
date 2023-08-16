import numpy as np

# Declare the different class types required for layers in the neural network
class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

class Dense(Layer):
    def __init__(self, output_size, input_size) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)

    def forward(self, input):
        self.input = input

        # Perform forward propagation multiplying weights by node values and adding a bias
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        '''
        Perform back propagation through the network through gradient descent, calculating the contribution
        of each node, weight and bias to the error through partial derivatives in an attempt at minimizing 
        the error. Each parameter is nudged accordingly to find a likely minimum of the cost function.
        E = Error, Y = Output value, W = weight, B = Bias
        This involves multipling dE/dY(Output gradient) by the inputs to those weights
        dE/dW = dE/dY * dY/dW --> dY/dW == Xi where X is the input node for that particular weight i.
        e.g Y1 = X1W11 + X2W12 + ... + XiW1i + b1
            dY1/dW12 = X2  hence dE/dW12 = dE/dY1 * X2
        dE/dBi = dE/dY * dYi/dBi since dBi is a constant, it evaluates to 1, hence:
        dE/dBi = dE/dY(Output gradient)
        dE/dXi = dE/dYj * dYj/dXi since dYj/dXi == Wji where j is the destination node of the neuron and i the origin, hence:
        dE/dXi = dE/dYj * summation(Wji) i.e all the weights resulting from the input node Xi to all output nodes Yj
        '''
        # Obtain gradient descent that will minimize cost function. Nudge parameters accordingly.
        weight_gradient = np.dot(output_gradient, self.input.T)

        # Calculate the gradient descent for the input to the current layer to serve as the output of the preceding layer
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Nudge parameters according to their contribution to the error. When value is positive you reduce it hence subtract, when negative you increase it hence add
        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient
    

class Activation(Layer):
    def __init__(self, activation, activation_derived) -> None:
        self.activation = activation
        self.activation_derived = activation_derived

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        '''
        Perform partial derivative of the activation function
        Yi = f(Xi)
        dE/dXi = dE/dYi * dYi/dXi since dYi/dXi = f'(Xi) then:
        dE/dXi = dE/dYi * f'(Xi) This is element-wise multiplication since each output has a corresponding activation function that produced it 
        hence we multiply every element in the output gradient(dE/dYi) matrix with its corresponding derivative f'(Xi)
        '''
        return np.multiply(output_gradient, self.activation_derived(self.input))
    
class Tanh(Activation):
    def __init__(self) -> None:
        def tanh(x):
            return np.tanh(x)
        
        def tanh_derived(x):
            return 1 - np.tanh(x) ** 2
        
        super().__init__(tanh, tanh_derived)

class Sigmoid(Activation):
    def __init__(self) -> None:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_derived(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_derived)


class Softmax(Layer):
    # Normalize output values to get a probability distribution with joint probabilities adding up to 1
    def forward(self, input):
        temp = np.exp(input)
        self.output = temp / np.sum(temp)

        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
    
