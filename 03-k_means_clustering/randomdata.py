import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def plot2d(x,y, colors=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y, c=colors)
    plt.show()

data, classes = make_blobs(n_samples=400, centers=randint(3,8))
plot2d(data[:,0], data[:,1])

# das beste K finden
scores = []

for i in range(1,11):
    print("Clustering with K=%d" % (i))
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    #scores.append([i, kmeans.score(data)])
    scores.append([i, mean_squared_distance(kmeans, data)])


kmeans = KMeans(n_clusters=4)
kmeans.fit(data)

pred = kmeans.predict(data)
plot2d(data[:,0], data[:,1], pred)





