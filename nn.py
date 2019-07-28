import numpy as np


class NeuralNetwork():

    def __init__(self, inputs, hiddens, outputs):
        layers = (inputs, hiddens, outputs)
        weightShapes = [(r, c) for r, c in zip(layers[1:], layers[:-1])]
        self.weightMatrices = [np.random.standard_normal(ws) for ws in weightShapes]
        self.biasVectors = [np.random.standard_normal((ls, 1)) for ls in layers[1:]]

    def info(self):
        print("weightMatrices ", self.weightMatrices)
        print("biasVectors ", self.biasVectors)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def diffsigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def predict(self, x):
        for weights, biases in zip(self.weightMatrices, self.biasVectors):
            x = self.sigmoid(weights @ x + biases)
        return x

    def loadTrainingData(self, path):
        self.trainingData = np.load(path)
        return self.trainingData

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
        gradients = (gradients @ outputErrors) * learningRate
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

# MAIN
net = NeuralNetwork(2, 4, 1)
data = net.loadTrainingData('XORdata.npz')

inputMat = data['x']
targetMat = data['y']


print("training...\n\n")
for i in range(4):
    for j in range(1000):
        net.train(0.01, i)
    x = np.reshape(inputMat[i], (2, 1))
    y = np.reshape(targetMat[i], (1, 1))
    print("\ninput:\n{0}\nprediction:{1}\ntarget:{2}\n".format(
        x, net.predict(x), y))

# for i in range(4):
#     x = np.reshape(inputMat[i], (2,1))
#     y = np.reshape(targetMat[i], (1,1))
#     print("input:\n{0}\nprediction:{1}\ntarget:{2}\n".format(x, net.predict(x), y))
