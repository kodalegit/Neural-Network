# Neural Network
## Description
This is a python implementation of a simple neural network. The neural network is modular in that layers can be added or removed as required. The neural network has achieved an 86% accuracy in predicting image labels from the MNIST dataset.

The following are the contents of each file in the project:
- layers.py -  Contains classes for layers required in the neural network. These include the dense layer class inheriting from the base layer class for implementing weights and biases for each layer as well as forward and back propagation, activation function classes inheriting from the base layer class that implement the tanh and sigmoid activation functions as well as forward and back propagation, and softmax layer that inherits from the base layer class that transforms network output into a probability distribution adding up to 1.
- mnist.py - Contains code for training and testing the neural network model on the mnist dataset and computing the accuracy of the model.

## Key Takeaways
1. The neural network performs gradient descent on each training example and thus is quite computationally expensive. A possible solution is implementation of a stochastic gradient descent where training examples are grouped into mini-batches and gradient descent computed using computations from the mini-batch. This can speed up convergence.
2. The network is susceptible to overfitting. This can be solved by implementation of dropout to encourage the network to develop multiple pathways for making predictions and thus improve its generalization.
3. Adding convolutional layers can improve the efficiency of the model in handling images without adding too much complexity to the architecture.
4. The gradient of the loss function(mean squared error) is obtained by the element-wise difference between the model's outputs and the corresponding true labels. This is found to improve training speed and accuracy as opposed to performing the derivative of the mean squared error to obtain the gradient of the loss function. The resulting gradient obtained using the applied method aligns more closely with the error between predicted and actual values potentially leading to smoother and more effective parameter updates during optimization. 

## Getting Started
The program is run from the command-line as such `python3 mnist.py`
    