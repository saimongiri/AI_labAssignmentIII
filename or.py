import random
import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[1]])

wt = np.random.rand(2)
b = np.random.rand(1)
lr = 0.1
for epoch in range(20) :
    te = 0
    for i in range(len(x)) :
        z = np.dot(x[i],wt) + b
        error = y[i] - z
        te += error**2
        wt += lr * error * x[i]
        b += lr * error
    print("Epoch {}: Total Error = {}".format(epoch+1, te[0]))
print("Final weights: ", wt)
print("Final bias: ", b)   

for i in range(len(x)) :
    z = np.dot(x[i],wt) + b
    output = 1 if z >= 0.5 else 0
    print("Input: {}, Output: {}".format(x[i], output))