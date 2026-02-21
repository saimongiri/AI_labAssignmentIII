import numpy as np
import random
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
np.random.seed(0)
wt1 = np.random.rand(2,2)
b1 = np.random.rand(1,2)
wt2 = np.random.rand(2,1)
b2 = np.random.rand(1,1)
lr = 0.1
def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z) :
    return z * (1 - z)
for epoch in range(20) :
    hidden_input = np.dot(x, wt1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wt2) + b2
    final_output = sigmoid(final_input)

    error = np.mean(np.square(y - final_output))
    d_output = (y-final_output) * sigmoid_derivative(final_output)
    d_hidden_output = d_output.dot(wt2.T) * sigmoid_derivative(hidden_output)
    
    wt2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    wt1 += x.T.dot(d_hidden_output) * lr
    b1 += np.sum(d_hidden_output, axis=0, keepdims=True) *lr

print("Final weights between input and hidden layer: ", wt1)
print("Final bias for hidden layer: ", b1)
print("Final weights between hidden and output layer: ", wt2)
print("Final bias for output layer: ", b2)
for i in range(len(x)) :
    hidden_input = np.dot(x[i], wt1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wt2) + b2
    final_output = sigmoid(final_input)
    output = 1 if final_output >= 0.76 else 0
    print("Input: {}, Output: {}".format(x[i], output))


