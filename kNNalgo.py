import numpy as np
from collections import Counter

def knn(X_train, y_train, X_test, k):
    predictions = []
    for test in X_test:
        distances = np.sqrt(np.sum((X_train - test)**2, axis=1))
        k_indices = distances.argsort()[:k]
        k_labels = y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

X_train = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])
y_train = np.array([0,0,0,1,1,1])

X_test = np.array([[2,2],[7,6]])

result = knn(X_train, y_train, X_test, k=3)

print(result)