# http://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

# Datensatz laden und Spalten benennen
df = pd.read_csv("fish.csv")
df.columns = ['alter', 'wassertemperatur', 'laenge']

# Daten plotten
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=df.alter, ys=df.wassertemperatur, zs=df.laenge)

# Trainings- und Testdaten
test = df.sample(int(len(df)*0.2))
train = df.drop(df.index[test.index])

# Lineare Regression mit scikit-learn
x_train = np.array(train[['alter', 'wassertemperatur']])
y_train = np.array(train.laenge)

x_test = np.array(test[['alter', 'wassertemperatur']])
y_test = np.array(test.laenge)

regr = linear_model.LinearRegression()

regr.fit(x_train, y_train)

print('Coefficients: ', regr.coef_)
print("Mean squared error: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
print('Variance score: %.2f' % regr.score(x_train, y_train))

ax.scatter(xs=x_test[:,0], ys=x_test[:,1], zs=regr.predict(x_test))

plt.show()
