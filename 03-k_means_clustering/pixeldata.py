import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import imread
from mpl_toolkits.mplot3d import Axes3D

# TODO: Translate comments to german

def main():
    data = load_image("wallpaper.jpg")
    print(data.shape)
    #plot3d(data[:,0], data[:,1], data[:,2])

    scores = []
    
    # ==============================================
    # Iteration to find the perfect K for clustering
    # and plot the results
    # ==============================================

    #for i in range(1,11):
    #    print("Clustering with K=%d" % (i))
    #    kmeans = KMeans(n_clusters=i)
    #    kmeans.fit(data)
    #    #scores.append([i, kmeans.score(data)])
    #    scores.append([i, mean_squared_distance(kmeans, data)])

    #scores = np.array(scores)
    #plot_cluster_score(scores[:,0], scores[:,1])

    # ============================================================
    # Iteration to find out how the score of the clustering varies
    # for the next 10 iterations
    # ============================================================

    #scores = []
    #for i in range(10):
    #    print("Running iteration %d" % (i))
    #    kmeans = KMeans(n_clusters=3)
    #    kmeans.fit(data)
    #    scores.append([i, mean_squared_distance(kmeans, data)])

    #scores = np.array(scores)
    #plot_cluster_score(scores[:,0], scores[:,1])
    
   
    # Plot data and color the different classes with the colors 
    # of the centers.
    print("Executing k-means for the last time")
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    
    # Reduce data, for faster plotting and faster panning
    data = np.unique(data, axis=0)
    print("Plotting data...")
    print(data.shape)
    print(np.unique(kmeans.labels_))

    cluster_colors = np.array([kmeans.cluster_centers_[x] for x in kmeans.labels_])
    #data = kmeans.cluster_centers_
    #cluster_colors = data
    plot3d(data[:,0], data[:,1], data[:,2], colors=cluster_colors/255)


# Einfache Funktion für Darstellung von dreidimensionalen Daten
def plot3d(x,y,z, colors=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, c=colors)
    plt.show()

def plot_cluster_score(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y,'-')
    plt.show()

def load_image(imagefile):
    img = imread(imagefile)
    newdata = []

    for x in range(len(img)):
        for y in range(len(img[x])):
            pic = img[x][y]
            newdata.append([pic[0], pic[1], pic[2]])
    return np.array(newdata)

def mean_squared_distance(clustering, data):
    distribution = clustering.predict(data)
    dst = 0

    # compute distance between data point and cluster center
    for i in range(len(distribution)):
        a = data[i]
        b = clustering.cluster_centers_[distribution[i]]

        dst += np.linalg.norm(a-b) ** 2

    return np.mean(dst)


if __name__ == "__main__":
    main()

