# Neural Network
## Description
This is a python implementation of a simple neural network. The neural network is modular in that layers can be added or removed as required.

The following are the contents of each file in the project:
- layers.py -  Contains classes for layers required in the neural network. These include the dense layer class inheriting from the base layer class for implementing weights and biases for each layer as well as forward and back propagation, activation function classes inheriting from the base layer class that implement the tanh and sigmoid activation functions as well as forward and back propagation, and softmax layer that inherits from the base layer class that transforms network output into a probability distribution adding up to 1.
- mnist.py - Contains code for training and testing the neural network model on the mnist dataset and computing the accuracy of the model.

## Challenges and potential improvements
1. The neural network performs gradient descent on each training example and thus is quite computationally expensive. A possible solution is implementation of a stochastic gradient descent where training examples are grouped into mini-batches and gradient descent computed from the average error in the random mini-batch. This can speed up convergence.
2. The network is susceptible to overfitting. This can be solved by implementation of dropout to encourage the network to develop multiple pathways for making predictions and thus improve its generalization.
3. Adding convolutional layers can improve the efficiency of the model in handling images without adding too much complexity to the architecture.

## Getting Started
The program is run from the command-line as such `python3 mnist.py`
    