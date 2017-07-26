# http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Einfache Funktion für einfache Datendarstellung
def plot2d(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    plt.show()


# Datensatz laden
data = datasets.load_boston()
df = pd.DataFrame(data.data)
df.columns = data.feature_names

target = pd.DataFrame(data.target)
target.columns = ["PRICE"]

# Numpy arrays für den Daten verwenden und Test- und Trainingsdaten erzeugen
x = np.array(df)
y = np.array(target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

regr = linear_model.LinearRegression(copy_X=True)
regr.fit(x_train, y_train)

# Koeffiziente der Regression
coef = pd.DataFrame({'features': df.columns, 'coef': regr.coef_[0]})
print(coef)

# interessante Daten plotten
#plot2d(df.NOX, target)
#plot2d(df.RM, target)
#plot2d(df.CHAS, target)

# Vorhersagen berechnen
pred = regr.predict(x_test)
#plot2d(y_test, pred)

# Mittlere Quadratfehler
mse = np.mean((y_test - regr.predict(x_test)) ** 2)
print(mse)

regr = linear_model.LinearRegression()
regr.fit(np.array(df.RM), y)
mse = np.mean((y_test - regr.predict(x_test)) ** 2)
print(mse)
