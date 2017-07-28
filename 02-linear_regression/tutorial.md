# Lineare Regression mit Python

Die lineare Regression ist ein statistisches Modell, das eine lineare Beziegung zwischen zwei oder mehr Variablen untersucht. Eine davon ist die abhängige Variable (auch erklärte gennant), die wir vorhersagen oder klassifizieren möchten. Wir haben auch mindestens eine unabhängige Varialbe (auch erklärende genannt).
Eine lineare Beziehung bedeutet formal, dass wir eine Funktion haben die so aussieht:

*Formel für lineares Modell hier*

*Erklärung der Notation*

Nicht formal bedeutet das, wenn der Wert einer der unabhängigen Variablen wächst, so wächst auch die abhängige Varialbe (oder sinkt, in diesem Fall sprechen wir von einer negativen linearen Beziehung). Unsere Aufgabe ist, wenn wir die Daten plotten, eine solche Linie zu ziehen, damit die mittlere quadratische Abstand von jeder Punkt zur Linie minimal ist. 
Das gilt natürlich für 2-dimensionale Daten. Für 3-Dimensionale Daten ist die Funktion keine Linie, sondern eine Ebene. 

*Beispiel*

Zusätzlich kann man ein lineares Modell auch für binäre Klassifizierung verwenden. Klassen werden typischerweise durch die Zahlen 0 und 1 dargestellt. Wegen der Natur des Algorithmus, bekommt man als Ergebnis vom Modell zahlen zwischen 0 und 1. Dann werden alle Zahlen größer als 0,5 auf 1 gesetzt und der Rest auf 0.

## Lineare Regression in Python

`Scikit-learn` ist ein mächtiges Machine Learning Framework. Das Modul `sklearn.linear_model` enthält viele Methoden die nützlich sind, wenn man eine abhängige Variable hat, die in lineares Verhältnis von den unabhängigen Variablen ist.
Außerdem werden wir `numpy` and `pandas` verwenden, die das Arbeiten mit Daten viel erleichtern, und `matplotlib` um Daten darzustellen.

### Datensatz

Für die Regressionsaufgabe werden wir einen Datensatz über Immobilenpreisen im Boston-Gebiet. Die Daten kommen ursprunglich vom Carnegie Mellon University und man kann ihn hier finden: [Link](http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) Zusätzlich sind die Daten in `sklearn.datasets` vorhanden.

Man kann entweder den Link öffnen oder `data.DESCR` aufrufen um eine Beschreibung des Datensatzes zu bekommen, die aber auf Englisch ist. 

Im Datensatz sind zwei Afgabe zu finden:
- Der PREIS des Houses zu berechnen (*Die Aufgabe, die wir in diesem Tutorial lösen*)
- Stickstoffmonoxidniveaus (NOX) vorhersagen 

### Implementierung

Zuerst importieren wir alle nötige Bilbiotheken.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
```

Wie gesagt, wir können diesmal den Datensatz vom sklearn importieren. Trotzdem speichern wir ihn in eine Pandas DataFrame, damit es einfacher ist, verschiedene Attribute zuzugreifen.
Die rückgabe von `load_boston()` ist ein Wörterbuch, mit 4 Attributen:
- `data.data` 
- `data.target`
- `data.feature_names`
- `data.DESCR`

```python
# Datensatz laden
data = datasets.load_boston()
df = pd.DataFrame(data.data)
df.columns = data.feature_names

target = pd.DataFrame(data.target)
target.columns = ["PRICE"]

```

### Klassifizierung mit dem linearen Modell
