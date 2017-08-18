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

def plot_cluster_score(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y,'-')
    plt.show()

data, classes = make_blobs(n_samples=400, centers=randint(3,8), random_state=42)
plot2d(data[:,0], data[:,1])

scores = []

# ==============================================
# Iteration to find the perfect K for clustering
# and plot the results
# ==============================================

for i in range(1,11):
    print("Clustering with K=%d" % (i))
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    scores.append([i, kmeans.score(data)])

scores = np.array(scores)
plot_cluster_score(scores[:,0], scores[:,1])

# ============================================================
# Iteration to find out how the score of the clustering varies
# for the next 10 iterations
# ============================================================

scores = []
for i in range(10):
    print("Running iteration %d" % (i))
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    scores.append([i, kmeans.score(data)])

scores = np.array(scores)
plot_cluster_score(scores[:,0], scores[:,1])

# =======================================================
# Do a clustering a final time using the best possible K,
# and plot the different clusters as different colors.
# =======================================================

print("Executing k-means for the last time")
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)

print("Plotting data...")
plot2d(data[:,0], data[:,1], kmeans.labels_)
