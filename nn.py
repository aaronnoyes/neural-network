#based on neural network described by jamesloyys
#https://gist.github.com/jamesloyys/ff7a7bb1540384f709856f9cdcdee70d#file-neural_network_backprop-py
#https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import pdb
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, x, y, activation='sigmoid', errorFunction='mse'):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.activation = activation
        self.errorFunction = errorFunction
        self.history = []

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x*(1-x)
        else:
            return 1 / (1 + np.exp(-x))

    def relu(self, x, derivative=False):
        if derivative:
            return 1. * (x > 0)
        else:
            return x * (x > 0)

    #calculate the mean squared error elementwise
    #axis None performs this element wise returning a scalar value
    def calculateError(self):
        return ((self.y - self.output)**2).mean(axis=None)

    def feedforward(self):
        if (self.activation == 'sigmoid'):
            self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
            self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

        if (self.activation == 'relu'):
            self.layer1 = self.relu(np.dot(self.input, self.weights1))
            self.output = self.relu(np.dot(self.layer1, self.weights2))

        return self.output

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        if (self.activation == 'sigmoid'):
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid(self.output, True)))
            d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid(self.output, True), self.weights2.T) * self.sigmoid(self.layer1, True)))

        if (self.activation == 'relu'):
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.relu(self.output, True)))
            d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.relu(self.output, True), self.weights2.T) * self.relu(self.layer1, True)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self):
        self.output = self.feedforward()
        self.history.append(self.calculateError())
        self.backprop()

    def learn(self):
        for i in range(1500):
            self.train()

    def summary(self):
        plt.plot(self.history)
        plt.xlabel('trials')
        plt.ylabel('mean squared error')
        print ("Input : \n" + str(self.input))
        print ("Actual Output : \n" + str(self.y))
        print ("Predicted Output: \n" + str(self.feedforward()))
        plt.show()

def main():
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format}) #show trailing 0s and always 5 points of precision for comparison
    x = np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
    y = np.array(([0],[1],[1],[0]), dtype=float)
    # nn = NeuralNetwork(x, y)
    NN = NeuralNetwork(x,y, 'sigmoid')
    NN.learn()
    NN.summary()

main()
