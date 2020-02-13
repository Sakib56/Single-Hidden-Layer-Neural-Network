# Single-Layer Neural Network (2019)

#### This is a very basic single hidden layer neural network built completely from scratch using python and numpy. 

This repo contains the neural network (nn.py), a way to generate the training/testing data (genXORdata.py) which will create the xor dataset (XORdata.npz) and a file that will link this all together by test the neural network and output results (exclusiveOR.py).

###### *If you're wondering how my neural network works, just open nn.py and read the comments I've written. I tried my best to make them as clear and understandable as possible.*

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

This will:
* initialise a new neural network (w/ 2 input neurons, 4 hidden neurons and 1 output neuron)
* generate and get the xor data
* train the neural network for 10000 iterations
* test and output the results

###### *Feel free to remove this code and mess around with the methods!*

Sample of expected output:
```python
...
training...

post-training tests
input:
[[0]
 [1]]
target:[[0]]
prediction:[[0.96355718]]
error:0.03644282488474359
...
```

## Why the XOR dataset?

The reason I chose to use the xor dataset for testing was because it is often considered as the “hello world” of machine learning (next to the mnist dataset). 

A neural network can be considered universal function approximators. Neural networks often have an activation function. This is basically a way to introduce non-linearity into the decision boundary so that we can classify datasets with more complex patterns. Without an activation function a neural net would only be able to classify linearly separable datasets e.g. datasets where you can draw *one* straight line to classify the points. 

for example, AND dataset is linearly separable whereas the XOR dataset is not...

| AND | 0 | 1 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 1 |

for AND, notice how we can draw *one* straight line to separate the 0s and 1s (cut off bottom right square to separate). Even a simple perceptron without an activation function could classify this.

| XOR | 0 | 1 |
| --- | --- | --- |
| 0 | 0 | 1 |
| 1 | 1 | 0 |

for XOR, now notice how there is no way to draw *one* straight line to separate the 0s and 1s. This will require the help of an activation function.
