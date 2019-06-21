# A Simple Neural Network hyperparameter comparison in Python

## What is this?
This is a I very basic neural network written in Python alond with a [paper](https://github.com/aaronnoyes/neural-network/blob/master/nn.pdf) which analyses the effects of several hyperparameters on the training of the network.


In the nn.py file I defined a method for comparing a list of these networks and displaying some graphical comparisons. I chose to create a learning method in the NeuralNetwork class that will train on the input data until the MSE reaches a certain threshold, and then added a limit so that if a particular network gets stuck at a local minima it will not hang.


## Credit
The guts of the neural network class are from [This blog post from jamesloyys on towardsdatascience.](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)
Their original code can be found [here](https://gist.github.com/jamesloyys/ff7a7bb1540384f709856f9cdcdee70d#file-neural_network_backprop-py)
