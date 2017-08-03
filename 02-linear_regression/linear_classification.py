import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Einfache Funktion für Darstellung von dreidimensionalen Daten
def plot3d(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    plt.show()

def plot_prediction3d(x1,y1,z1, x2,y2,z2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1,y1,z1, alpha=0.5)
    ax.scatter(x2,y2,z2)
    plt.show()

def fix_predictions(vector, threshold=0.5):
    for i in range(len(vector)):
        if vector[i] > 0.5:
            vector[i] = 1
        else:
            vector[i] = 0

    return vector

def classification_score(y, y_hat):
    correct = np.sum(y == y_hat)
    return correct/len(y)
    
#http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
# Zufallsdaten erzeugen
X,y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0)
plot3d(X[:,0], X[:,1], y)

# Lineare Regression erzeugen und anpassen
regr = LinearRegression()
regr.fit(X,y)

y_hat = fix_predictions(regr.predict(X))

print("Klassifikationsgüte vor Anpassung: %f: " % classification_score(y, regr.predict(X)))
plot_prediction3d(X[:,0], X[:,1], y, X[:,0], X[:,1], regr.predict(X))

print("Klassifikationsgüte nach Anpassung: %f: " % classification_score(y, y_hat))
plot_prediction3d(X[:,0], X[:,1], y, X[:,0], X[:,1], y_hat)


