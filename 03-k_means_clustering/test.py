import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import imread

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
            newdata.append([x,y,pic[0], pic[1], pic[2]])
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

def main():
    data = load_image("wallpaper.jpg")
    print(data.shape)

    scores = []

    for i in range(1,11):
        print("Clustering with K=%d" % (i))
        kmeans = KMeans(n_clusters=8)
        kmeans.fit(data)
        #scores.append([i, kmeans.score(data)])
        scores.append([i, mean_squared_distance(kmeans, data)])

    scores = np.array(scores)
    plot_cluster_score(scores[:,0], scores[:,1])


if __name__ == "__main__":
    main()

