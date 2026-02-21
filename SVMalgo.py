import numpy as np
from sklearn import svm

X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])
y = np.array([0,0,0,1,1,1])

model = svm.SVC(kernel='linear')
model.fit(X, y)

X_test = np.array([[2,2],[7,6]])

pred = model.predict(X_test)

print(pred)