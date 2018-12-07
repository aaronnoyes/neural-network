#based on neural network described by jamesloyys
#https://gist.github.com/jamesloyys/ff7a7bb1540384f709856f9cdcdee70d#file-neural_network_backprop-py
#https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import pdb

class NeuralNetwork:
    def __init__(self, x, y, activation='sigmoid', errorFunction='mse', learningRate=0.01, goalError=0.002, trialLimit=10000, minimizer='gd'):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.activation = activation
        self.errorFunction = errorFunction
        self.history = []
        self.learningRate = learningRate
        self.trials = 0
        self.goalError = goalError
        self.trialLimit = trialLimit
        self.minimizer = minimizer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x*(1-x)

    def relu(self, x):
        return x * (x > 0)

    def d_relu(self, x):
        return 1 * (x > 0)

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return (1 - (x ** 2))

    def softplus(self, x):
        return np.log(1+np.exp(x))

    def d_softplus(self, x):
        return 1 / (1 + np.exp(-x))

    #calculate the mean squared error elementwise
    #axis None performs this element wise returning a scalar value
    def calculateError(self):
        return ((self.y - self.output)**2).mean(axis=None)

    def feedforward(self):
        if (self.activation == 'sigmoid'):
            # pdb.set_trace()
            self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
            self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

        elif (self.activation == 'relu'):
            self.layer1 = self.relu(np.dot(self.input, self.weights1))
            self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

        elif (self.activation == 'tanh'):
            self.layer1 = self.tanh(np.dot(self.input, self.weights1))
            self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

        elif (self.activation == 'softplus'):
            self.layer1 = self.softplus(np.dot(self.input, self.weights1))
            self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

        return self.output

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        if (self.activation == 'sigmoid'):
            # pdb.set_trace()
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.d_sigmoid(self.output)))
            d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.d_sigmoid(self.output), self.weights2.T) * self.d_sigmoid(self.layer1)))

        elif (self.activation == 'relu'):
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.d_sigmoid(self.output)))
            d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.d_relu(self.output), self.weights2.T) * self.d_relu(self.layer1)))

        elif (self.activation == 'tanh'):
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.d_sigmoid(self.output)))
            d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.d_tanh(self.output), self.weights2.T) * self.d_tanh(self.layer1)))

        elif (self.activation == 'softplus'):
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.d_sigmoid(self.output)))
            d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.d_softplus(self.output), self.weights2.T) * self.d_softplus(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += self.learningRate * d_weights1
        self.weights2 += self.learningRate * d_weights2

    def train(self):
        if self.minimizer == 'gd':
            self.gradientDescent()
        elif self.minimizer == 'sgd':
            self.stochrasticGradiendDescent()

    def gradientDescent(self):
        self.output = self.feedforward()
        self.history.append(self.calculateError())
        self.backprop()

    def stochrasticGradiendDescent(self):
        print('TODO')

    def learn(self):
        while self.calculateError() >= self.goalError:
            self.trials += 1
            self.train()
            if self.trials >= self.trialLimit:
                print(self.activation + " timed out after " + str(self.trialLimit) + " trials\n")
                break

    def summary(self):

        print (self.activation)
        print ("Input : \n" + str(self.input))
        print ("Actual Output : \n" + str(self.y))
        print ("Predicted Output: \n" + str(self.feedforward()))
        print ("Mean Squared Error: " + str(self.calculateError()))
        print ("Trials Required: " + str(self.trials))
        print ("\n")
