import numpy as np

class NeuralNetwork():
    def __init__(self, inputs, hiddens, outputs):
        layers = (inputs, hiddens, outputs)
        weightShapes = [(r, c) for r, c in zip(layers[1:], layers[:-1])]
        self.weightMatrices = [np.random.standard_normal(ws) for ws in weightShapes]
        self.biasVectors = [np.random.standard_normal((ls, 1)) for ls in layers[1:]]

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
        inputMat = self.trainingData['x']
        targetMat = self.trainingData['y']

        weightsIH = self.weightMatrices[0]
        biasH = self.biasVectors[0]
        weightsHO = self.weightMatrices[1]
        biasO = self.biasVectors[1]

        inputs = np.reshape(inputMat[i], (2, 1))
        hiddens = self.sigmoid(weightsIH @ inputs + biasH)
        outputs = self.sigmoid(weightsHO @ hiddens + biasO)
        targets = np.reshape(targetMat[i], (1, 1))

        outputErrors = targets - outputs
        gradients = self.diffsigmoid(outputs)
        gradients = (gradients * outputErrors) * learningRate
        deltaWeightsHO = gradients @ np.transpose(hiddens)

        weightsHO += deltaWeightsHO
        biasO += gradients

        hiddenErrors = np.transpose(weightsHO) @ outputErrors
        hiddenGradients = self.diffsigmoid(hiddens)
        hiddenGradients = (hiddenGradients * hiddenErrors) * learningRate
        deltaWeightsIH = hiddenGradients @ np.transpose(inputs)

        weightsIH += deltaWeightsIH
        biasH += hiddenGradients

        self.weightMatrices[0] = weightsIH
        self.biasVectors[0] = biasH
        self.weightMatrices[1] = weightsHO
        self.biasVectors[1] = biasO