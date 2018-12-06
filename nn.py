#based on neural network described by jamesloyys
#https://gist.github.com/jamesloyys/ff7a7bb1540384f709856f9cdcdee70d#file-neural_network_backprop-py
#https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import pdb
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, x, y, activation='sigmoid', errorFunction='mse', learningRate=0.01, goalError=0.002, trialLimit=3000):
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
            self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

        return self.output

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        if (self.activation == 'sigmoid'):
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid(self.output, True)))
            d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid(self.output, True), self.weights2.T) * self.sigmoid(self.layer1, True)))


        if (self.activation == 'relu'):
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid(self.output, True)))
            d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.relu(self.output, True), self.weights2.T) * self.relu(self.layer1, True)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += self.learningRate * d_weights1
        self.weights2 += self.learningRate * d_weights2

    def train(self):
        self.output = self.feedforward()
        self.history.append(self.calculateError())
        self.backprop()

    def learn(self):
        while self.calculateError() > self.goalError:
            self.trials += 1
            self.train()
            if self.trials > self.trialLimit:
                print("Timed out after " + str(self.trialLimit) + " trials")
                break

    def summary(self):

        print (self.activation)
        print ("Input : \n" + str(self.input))
        print ("Actual Output : \n" + str(self.y))
        print ("Predicted Output: \n" + str(self.feedforward()))
        print ("Mean Squared Error: " + str(self.calculateError()))
        print ("Trials Required: " + str(self.trials))
        print ("\n")

def compareNetworks(networks):
    length = len(networks)
    trials = []
    activations = []
    plotColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] #used to match first sublpot colors

    #graph lines as errors change
    plt.xlabel('trials')
    plt.ylabel('mean squared error')
    for nn in networks:
        plt.subplot(211)
        plt.xlabel('trials')
        plt.ylabel('mean squared error')
        plt.plot(nn.history, label=nn.activation)
        trials.append(nn.trials)
        activations.append(nn.activation)
        nn.summary()
    plt.legend()

    #bar chart for activation trials required
    plt.subplot(212)
    plt.xlabel('trials')
    plt.legend()
    plt.bar(np.arange(len(networks)), trials, align='center', color=plotColors)
    plt.xticks(np.arange(len(networks)), activations)

    #clean up spacing between subplots
    plt.tight_layout()
    plt.show()

def main():
    #make numpy print same precision for all values
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format}) #show trailing 0s and always 5 points of precision for comparison

    x = np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float) #nn inputs
    y = np.array(([0],[1],[1],[0]), dtype=float) #expected outputs
    networks = [] #hold all neural networks for comparison

    #sigmoid network
    NN = NeuralNetwork(x,y, activation='sigmoid', learningRate=1)
    NN.learn()
    networks.append(NN)

    #relu network
    NN2 = NeuralNetwork(x,y, activation='relu', learningRate=0.01)
    NN2.learn()
    networks.append(NN2)
    
    compareNetworks(networks)

main()
