import numpy as np

def kmeans(X, k, epochs=100):
    centroids = X[:k]
    
    for _ in range(epochs):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
        
    return centroids, labels

X = np.array([[1,2],[1,4],[1,0],
              [10,2],[10,4],[10,0]])

centroids, labels = kmeans(X, k=2)

print("Centroids:")
print(centroids)
print("Labels:")
print(labels)