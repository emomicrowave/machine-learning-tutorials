import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

# Datensatz laden 
df = pd.read_csv('iris.csv', header=None)
df.columns = ['sepalum_laenge', 'sepalum_breite', 'petalum_laenge', 'petalum_breite', 'klasse']

# Klasse der Datensatz als numerische Werte darstellen
df['klasse'] = pd.factorize(df['klasse'])[0]
print(df.head(5))

# Daten plotten
colors = list(df['klasse'])
plt.scatter(df['sepalum_laenge'], df['sepalum_breite'], c=colors)
plt.title("Sepalum Länge und Sepalum Breite")
plt.xlabel("Sepalum Länge")
plt.ylabel("Sepalum Breite")
plt.show()

plt.scatter(df['petalum_laenge'], df['petalum_breite'], c=colors)
plt.title("Petalum Länge und Petalum Breite")
plt.xlabel("Petalum Länge")
plt.ylabel("Petalum Breite")
plt.show()

# Training- und Testdaten erzeugen
test = df.sample(int(len(df) * 0.2))
train = df.drop(df.index[test.index])

# kNN Implementierung für Klassifizierung benutzen

## In numpy Arrays konvertieren
x_train = np.array(train.ix[:, 0:4])
y_train = np.array(train['klasse'])

x_test = np.array(test.ix[:, 0:4])
y_test = np.array(test['klasse'])

## kNN Algorithmus

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

pred = knn.predict(x_test)

### Funktion, die die Genauigkeit der Vorhersage berechnet
def calculate_accuracy(pred, target):
    counter = 0

    for i in range(len(pred)):
        if pred[i] == target[i]:
            counter += 1

    return counter / len(pred)


print(calculate_accuracy(pred, y_test))

## eigenen kNN Algorithmus erstellen
def predict_one(k_neighbors, x_train, y_train, to_predict):
    
    distances = []
    neighbors = []

    # alle Abstände holen als (index, Vektor) speichern
    for i in range(len(x_train)):
        dst = np.linalg.norm(x_train[i] - to_predict)
        distances.append((i, dst))

    # liste nach Abstand sortieren
    distances.sort(key=lambda tup: tup[1])

    # k Nachbarn holen
    for i in range(k_neighbors):
        neighbors.append(y_train[distances[i][0]])

    # Ausgabe ist am meisten vorkommender Element
    return max(set(neighbors), key=neighbors.count)

def kNN(k_neighbors, x_train, y_train, x_test):

    predictions = []

    for vec in x_test:
        predictions.append(predict_one(k_neighbors, x_train, y_train, vec))
    
    return predictions

pred = kNN(3, x_train, y_train, x_test)
print(calculate_accuracy(pred, y_test))
