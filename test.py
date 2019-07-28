import numpy as np

import nn
net = nn.NeuralNetwork(2, 4, 1)
data = net.loadTrainingData('XORdata.npz')
inputMat = data['x']
targetMat = data['y']

import random
print("\ntraining...\n")
for j in range(50000):
    net.train(0.01, random.choice([0,1,2,3]))
print("training complete!\n")

print("post-training tests\n")
for i in range(4):
    x = np.reshape(inputMat[i], (2,1))
    y = np.reshape(targetMat[i], (1,1))
    preds = net.predict(x)
    err = abs(y-net.predict(x))
    print("\ninput:\n{0}\nprediction:{1}\ntarget:{2}\nerror:{3}".format(x, preds, y, err))