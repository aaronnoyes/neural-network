#program for comparing neural networks

import numpy as np
import sys
import pdb
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

def compareNetworks(networks):
    length = len(networks)
    trials = []
    activations = []
    plotColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] #used to match first sublpot colors

    #graph lines as errors change
    plt.subplot(211)
    plt.xlabel('trial')
    plt.ylabel('mean squared error')
    for nn in networks:
        plt.plot(nn.history, label=nn.activation)
        trials.append(nn.trials)
        activations.append(nn.activation)
        nn.summary()
    plt.legend(loc="upper right")

    #bar chart for activation trials required
    plt.subplot(212)
    plt.xlabel('trials required')
    plt.ylabel('activation function')
    plt.barh(np.arange(len(networks)), trials, align='center', color=plotColors)
    plt.yticks(np.arange(len(networks)), activations)
    for i, v in enumerate(trials):
        plt.text(v*.5, i, str(v), color='black')

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
    NN = NeuralNetwork(x,y, activation='sigmoid', learningRate=0.3)
    NN.learn()
    networks.append(NN)

    #relu network
    NN2 = NeuralNetwork(x,y, activation='relu', learningRate=0.3)
    NN2.learn()
    networks.append(NN2)

    #relu network
    NN3 = NeuralNetwork(x,y, activation='tanh', learningRate=0.3)
    NN3.learn()
    networks.append(NN3)

    compareNetworks(networks)

main()
