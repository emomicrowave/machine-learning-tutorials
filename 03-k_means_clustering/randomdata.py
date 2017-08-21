import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

K_PARAM = 2

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

data, classes = make_blobs(n_samples=1000, centers=randint(3,8), random_state=42)
plot2d(data[:,0], data[:,1])

# Iteration um das perfekte K zu finden 
# und das Ergebnis zu plotten
scores = []
for i in range(1,11):
    print("Clustering mit K=%d" % (i))
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data)
    scores.append([i, kmeans.score(data)])

# einfacher Daten zu plotten wenn sie ein numpy-array sind, aber
# einfacher Daten in einer python-Liste hinzufügen
scores = np.array(scores)
plot_cluster_score(scores[:,0], scores[:,1])

# Iterieren um festzustellen, wie sich die Summe der Abstände
# sich für die nächste 10 Iteration verändert
scores = []
for i in range(10):
    print("Running iteration %d" % (i))
    kmeans = KMeans(n_clusters=K_PARAM)
    kmeans.fit(data)
    scores.append([i, kmeans.score(data)])

scores = np.array(scores)
plot_cluster_score(scores[:,0], scores[:,1])

# Clustering zum letzten mal Ausführen und
# Endergebnis in unterschiedliche Farben plotten
print("Executing k-means for the last time")
kmeans = KMeans(n_clusters=K_PARAM)
kmeans.fit(data)

print("Plotting data...")
plot2d(data[:,0], data[:,1], kmeans.labels_)
