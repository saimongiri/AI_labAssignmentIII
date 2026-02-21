import numpy as np

X = np.array([[0],[1]])
y = np.array([1,0])

w = np.random.rand(1)
b = np.random.rand(1)
lr = 0.1

def act(x):
    return 1 if x >= 0 else 0

for _ in range(10):
    for i in range(len(X)):
        y_pred = act(np.dot(X[i], w) + b)
        e = y[i] - y_pred
        w += lr * e * X[i]
        b += lr * e

for i in range(len(X)):
    print(act(np.dot(X[i], w) + b))