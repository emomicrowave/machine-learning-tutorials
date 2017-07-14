# http://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Datensatz laden und Spalten benennen
df = pd.read_csv("fish.csv")
df.columns = ['alter', 'wassertemperatur', 'laenge']

# Daten plotten
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=df.alter, ys=df.wassertemperatur, zs=df.laenge)

plt.show()



