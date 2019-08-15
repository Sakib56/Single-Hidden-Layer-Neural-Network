# Single-Layer Neural Network

This is a very basic single hidden layer neural network built completely from scratch using python and numpy. 

This repo contains the neural network (nn.py), the xor dataset (XORdata.npz) and a way to generate that dataset (genXORdata.py).

## How to run?
Just clone the repo into a folder and open a terminal in that directory. To train and test the neural network given the xor dataset, just run ```python exclusiveOR.py```

By default the following code will run...
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
>>> import genXORdata as xor

>>> net = nn.NeuralNetwork(2, 4, 1)
>>> xor.getData()
data = net.loadTrainingData('XORdata.npz')
learningRate = 0.1
trainingIter = 10000
xor.teachModel()
xor.testModel()
```
