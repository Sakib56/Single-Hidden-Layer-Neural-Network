# Single-Layer Neural Network

#### This is a very basic single hidden layer neural network built completely from scratch using python and numpy. 

This repo contains the neural network (nn.py), a way to generate the training/testing data (genXORdata.py) which will create the xor dataset (XORdata.npz) and a file that will link this all together by test the neural network and output results (exclusiveOR.py).

## How to run?
Just clone the repo into a folder and open a terminal in that directory. To train and test the neural network given the xor dataset, just run ```python exclusiveOR.py```

By default, the following code will run...
```python
### MAIN ###
net = nn.NeuralNetwork(2, 4, 1)
getData()
data = net.loadTrainingData('XORdata.npz')
learningRate = 0.1
trainingIter = 10000
teachModel()
testModel()
```

Alternatively, you can open a terminal in the directory and just import the relevant modules and run the methods.
For example:
```python
python
Python 3.6.5 (v3.6.5:f59c0932b4, Mar 28 2018, 17:00:18) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import nn
>>> import exclusiveOR as xor

>>> net = nn.NeuralNetwork(2, 4, 1)
>>> xor.getData()
data = net.loadTrainingData('XORdata.npz')
learningRate = 0.1
trainingIter = 10000
xor.teachModel()
xor.testModel()
```

## Why the XOR dataset?

The reason I chose to use the xor dataset for testing was because it is often considered as the “hello world” of machine learning (next to the mnist dataset). 

A neural network can be considered universal function approximators. Neural networks often have an activation function. This is basically a way to introduce non-linearity into the decision boundary so that we can classify datasets with more complex patterns. Without an activation function a neural net would only be able to classify linearly separable datasets e.g. and datasets (since you can draw *one* straight line and classify them).

| AND | 0 | 1 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 1 |

notice how we can draw *one* straight line to seperate the 0s and 1s (cut off bottom right square to seperate). Even a simple perceptron without an activation function could classify this.

| XOR | 0 | 1 |
| --- | --- | --- |
| 0 | 0 | 1 |
| 1 | 1 | 0 |

now notice how there is no way to draw *one* straight line to seperate the 0s and 1s. This will require the help of an activation functions (I use sigmoid in this repo).
