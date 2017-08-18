## Key points
- unsupervized learning
- uses of clustering
- how does it work?
- how to choose a K?
- clustering on the Iris Dataset
- artificial dataset from [here](https://github.com/deric/clustering-benchmark)
- **NO** clustering on image pixel data - *data not suited for k-means clustering*
- **NEEDS MORE RESEARCH** feature engineering
- different runs, different results (*spherical dataset*)
- **NEEDS MORE RESEARCH** what is validation

## TODOS:
- find a suitable dataset (*IRIS might be one*)
- implementation: choose a K
- implementation: clustering with chosen K
- save `matplotlib` plots
- use *Key points* structure to write tutorial text
- clustering on image pixel data?
- generate clustering steps images for the tutorial

# Einführung

Das ist der dritte Teil einer Serie von Blogbeiträge über Machine learning mit Python. In diesem Tutorial, beschäftigen wir uns mit unbeaufsichtigten Lernen 
und werden Clusteranalyse mit dem K-means Algorithmus implementieren. Grundkenntnisse in Python sind empfohlen. Außerdem ist es empfohlen sich die vorherigen Tutorials anzusehen. 

## Was ist Clusteranalyse

Clusteranalyse ist ein unbeaufsichtigtes Verfahren in den Machine Learning und Data Mining, wobei man Gruppen von ähnlichen Objekten in unbeschrifteten Daten finden will. 
Diese Art von Analyse hat Anwendungen in verschiedenen Bereichen wie Marketing (neue Kundengruppen auf basis ihrer Einkäufe zu identifizieren), 
Biologie (Klassifizierung von Individuen oder Gen-Sequenzierung), Sozialwissenschaften (Kriminalität Hotspots identifizieren) und viele andere.


## K-Means

K-Means steht für K-Mittelwerte und ist ein iterativer Algorthmus für Clusteranalyse. Als Eingabe bekommt der Algorithmus eine Zahl *K* und den Datensatz (Sammlung von Merkmale für jede Stichprobe) 
und versucht danach K unterschiedliche Gruppen im Datensatz zu erkennen. 

### Initialisierung

Die erste Schritt ist die Zentroiden der Clustern zu initialisieren. Am häufigsten nimmt man entweder K zufällige Stichproben aus dem Datensatz, oder K zufällige Punkte dazwischen. 

### Verteilung der Daten unter den Clustern

Die nächste Schritt jede Stichprobe in einen Cluster hinzufügen. Am häufigsten verwendet man der quadrat der euklidischen Abstand für eine Metrik. Das heißt es wird das nächstliegende Zentroid gewählt.

### Aktualisierung der Mittelwerte

Zunächst werden die neuen Mittelwerte für jeden Cluster berechnet.

### Iteration

Die letzten zwei Schritte werden so viel wiederholt bis der Algorithmus ein Stoppkriterium erreicht. Das kann eine Obergrenze der Iterationen sein, oder eine Untergrenze der Veränderung der Koordinaten der Zentroiden. Der Algorithmus konvergiert immer,
es kann aber sein, dass das keine optimale Lösung ist, und man muss mehrfach den Alorithmus ausführen.

## Implementierung mit Python und sklearn


Für die Implemetierung verwenden wir die folgenden Python-Bibliotheken:
- `sklearn` und `numpy` für algebraische Berechnungen und für eine Implementierung von K-means
- `matplotlib` für dreidimensionalen Datendarstellung
- `scipy` für die `imread` Funktion, die ein Bild in einer numpy-array umwandelt

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import imread
from mpl_toolkits.mplot3d import Axes3D
```

## Validierung

## Zusammenfassung

## Resourcen









