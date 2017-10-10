# Dataset:
# http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Einfache Funktion für einfache Datendarstellung
def plot2d(x, y, plot_title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    if plot_title:
        plt.title(plot_title)
    plt.show()

def mean_squared_error(y,y_hat):
    return np.mean((y - y_hat) ** 2)

# Datensatz laden
data = datasets.load_boston()
df = pd.DataFrame(data.data)
df.columns = data.feature_names

target = pd.DataFrame(data.target)
target.columns = ["PRICE"]

# Numpy arrays für den Daten verwenden und Test- und Trainingsdaten erzeugen
x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.25, random_state=42)

# Anpassen des linearen Modells
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

# Koeffiziente der Regression
coef = pd.DataFrame({'features': df.columns, 'coef': regr.coef_[0]})
print(coef)

# interessante Daten plotten
plot2d(df.NOX, target, "Verhältnis zwischen NOX (Konzentration von Stockstoffmonoxid)\n und den Preis der Wohnungen")
plot2d(df.RM, target, "Verhältnis zwischen RM (Durchschnittliche Anzahl von Zimmer)\n und den Preis der Wohnungen")
plot2d(df.CHAS, target, "Verhältnis zwischen CHAS (Wohnung an Charles River)\n und den Preis der Wohnungen")

# Vorhersagen berechnen
pred = regr.predict(x_test)
plot2d(y_test, pred)

# Mittlere Quadratfehler
mse = mean_squared_error(y_test, regr.predict(x_test))
print("Mittlere Quadratfehler: %f" % mse)

# alternative function: 
# regr.score(x_test, y_test)

# mit einem anderen Modell für Lineare Regression versuchen
new_X = df[["RM", "NOX"]]

regr = linear_model.LinearRegression()
regr.fit(new_X, target)

mse = mean_squared_error(target, regr.predict(new_X))
print("Mittlere Quadratfehler: %f" % mse)


