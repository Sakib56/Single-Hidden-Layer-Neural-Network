import numpy as np


class NeuralNetwork():
    def __init__(self, inputs, hiddens, outputs):
        layers = (inputs, hiddens, outputs)
        weightShapes = [(r, c) for r, c in zip(layers[1:], layers[:-1])]
        self.weightMatrices = [np.random.standard_normal(ws)/ws[1]**0.5 for ws in weightShapes]
        self.biasVectors = [np.zeros((ls, 1)) for ls in layers[1:]]

    # prints current weights and biases
    def info(self):
        print("weightMatrices ", self.weightMatrices)
        print("biasVectors ", self.biasVectors)

    # activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # differentiated activation function
    def diffsigmoid(self, x):
        # no need to run through sigmoid again since x has already been through activation
        return x * (1 - x)

    # feedforward algorithm
    def predict(self, x):
        for weights, biases in zip(self.weightMatrices, self.biasVectors):
            x = self.sigmoid(weights @ x + biases)
        return x

    # loads .npz containing training/testing data
    def loadTrainingData(self, path):
        self.trainingData = np.load(path)
        return self.trainingData

    # backpropagation algorithm
    def train(self, learningRate, i):
        # learningRate = step size in gradient decent, i = row in self.trainingData

        # exctaring "x_train" and "y_train"
        inputMat = self.trainingData['x']
        targetMat = self.trainingData['y']

        # getting weight matrix and bias vector for each layer
        weightsIH = self.weightMatrices[0]
        biasH = self.biasVectors[0]
        weightsHO = self.weightMatrices[1]
        biasO = self.biasVectors[1]

        # generating outputs for each layer (hiddens and outputs)
        inputs = np.reshape(inputMat[i], (2, 1))
        hiddens = self.sigmoid(weightsIH @ inputs + biasH)
        outputs = self.sigmoid(weightsHO @ hiddens + biasO)
        targets = np.reshape(targetMat[i], (1, 1))

        # calculating the output error, targets - outputs
        # calculating output gradient, m = s(o)*(1 - s(o))
        # hadamard matrix product used (entrywise product)
        # calculating output deltas, using prior layer; hiddens
        outputErrors = targets - outputs
        outputGradients = self.diffsigmoid(outputs)
        outputGradients = (outputGradients * outputErrors) * learningRate
        deltaWeightsHO = outputGradients @ np.transpose(hiddens)

        # adjusting the weights and bias vectors by deltas
        weightsHO += deltaWeightsHO
        biasO += outputGradients

        # calculating the hidden error, based on output errors
        # calculating hidden gradient, m = s(h)*(1 - s(h))
        # hadamard matrix product used (entrywise product)
        # calculating hidden deltas, using prior layer; inputs
        hiddenErrors = np.transpose(weightsHO) @ outputErrors
        hiddenGradients = self.diffsigmoid(hiddens)
        hiddenGradients = (hiddenGradients * hiddenErrors) * learningRate
        deltaWeightsIH = hiddenGradients @ np.transpose(inputs)

        # adjusting the weights and bias vectors by deltas
        weightsIH += deltaWeightsIH
        biasH += hiddenGradients

        # updating object's copy of matrices and vectors
        self.weightMatrices[0] = weightsIH
        self.biasVectors[0] = biasH
        self.weightMatrices[1] = weightsHO
        self.biasVectors[1] = biasO