## Notes and ideas

- simple use cases
- implementation with scikit learn
- pros and cons
- SVN Parameter in scikit-learn
- plot decision boundries
- plot support vectors

# Support Vector Machine (SVN)
Support Vector Machine ist ein überwachtes Machine Learning Verfahren, überwiegend für binäre Klassifizierung benutzt.
Die Trainingsdaten werden im n-dimensionalen Raum geplottet und der Algorithmus versucht eine solche Grenze zwischen
den beiden Klassen zu finden, so dass ein möglichst breiter Bereich frei von Stichproben um die Grenze herum ensteht.

### Stützvektoren
Die Grenze ist mittels der nächstliegenden Objekte definiert, die man deswegen *Stützvektoren* nennt. Dafür sind Vektoren, 
die von der Grenze entfernt liegen, beeinflussen ihre Berechnung nicht, deswegen brauchen die auch nicht im Hauptspeicher
zu stecken. Aus dieser Grund sind SVN Speichereffizient.

### Hyperebenen
Wenn man um Grenzen spricht, Das ist ein Unterraum, dessen Dimension um 1 kleiner ist als seine Umgebung. 
Zum Beispiel im dreidimensionalen Raum, wäre eine Hyperebene eine zweidimensionale Ebene. Und im zweidimensionalen Raum wäre
eine Hyperebene einfach eine gerade Linie. 

### Kernel-Trick
Daraus folgt aber, dass die Daten linear trennbar sind, was für die meisten reellen Fälle nicht der Fall ist. 
Deswegen kann man den sogenannten *Kernel-Trick* verwenden. Die Idee dahinter ist das Vektorraum in einem höherdimensionalen 
Raum zu überführen, wobei die Objekte schon linear trennbar sind und da eine Hyperebene zu definieren. 
Bei der Rücktransformation, wird diese Hyperebene nichtlinear und oft auch nicht zusammenhängend.

Probleme sind der hohe Rechenaufwand bei der Dimensionsteigerung und die oft nicht intuitive und unbrauchbare Form der
Hyperebene nach der Dimensionsreduktion. *kernel functions here*

**TODO: the heck are kernel fucntions**

## Implementierung mit Python

Wir nehmen den [Irisdatensatz](https://en.wikipedia.org/wiki/Iris_flower_data_set) und als Bibliotheken benutzen wir
`sklearn` und `numpy`. 

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Datensatz laden
data = load_iris()
features = np.array(data.data)
labels = np.array(data.target)
```

Typischerweise kann man ein SMV Konstruktur bilden mit `svm = SVC()`. `SVC` steht für *Support Vector Classificator* und im Gegensatz zu *Support Vector Regressor* wird nur für Klassifikationsprobleme verwendet. Mit `svm.fit(features, labels)` trainiert man das Modell und mit `svm.predict()` kann man die Vorhersagen für neue Stichprobe erhalten. 

Interessant bei der Implementierung von SVM sind die Kernel-funktionen und die Konstruktorparameter, die man anpassen kann. 
Wir werden diese Parameter erläutern und die Unterschiede visualisieren. Dafür brauchen wir eine neue funktion zu definieren.

```python
def plot_stuff(X, y, kernel_fn="linear", C=1.0, degree=3, gamma='auto'):
    svm = SVC(kernel=kernel_fn, C=C, degree=degree, gamma=gamma)
    svm.fit(X,y)
    
    # Minimale und Maximale Werte aus X nehmen und ein
    # Matrix bilden mit alle Koordinaten mit Abstand 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.01))
    
    # Vorhersagen nehmen für alle Koordinaten
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Hintergrund färben für jede Punkt aus xx,yy
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.summer)
    plt.axis('off')
	plt.title("Kernel Function: %s \n C: %.2f \n gamma: %s " % (kernel_fn, C, str(gamma)))

    # Daten auch als Punkte plotten
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.summer, edgecolors='black')
    plt.show()
```

### Kernel-Funktion


### Gamma


### Strafparameter (C)


## Ressourcen
- Logistische Regression vs. Decision Trees vs. Support Vector Machines - [Link](https://www.edvancer.in/logistic-regression-vs-decision-trees-vs-svm-part1/)
