# Einführung in der Machine Learning mit Python

Das ist der erste von einer Serie von Blogbeiträge über Machine Learning. Wir werden die Programmiersprache Python verwenden. Die Blogbeiträge gehen davon aus, dass die Leser keine Statistikshintergrund haben, haben aber Grundkenntnisse in Python und können Python-Bibliotheken installieren.

## Was ist Machine Learning?

Machine Learning ist ein Gebiet der künstlichen Intelligenz, das als Ziel hat, solche Computerprogramme zu entwickeln, die selbständig von Daten lernen können, und aufgrund der Daten, präzise Vorhersagen oder Klassifizierunge zu liefern. Typische Probleme aus dem Bereich der Machine Learning sind:

- Der Preis von Aktien nach 6 Monaten vorhersagen, basiert auf Unternehmenleistungen und Wirtschaftsdaten.
- Die Ziffer einer Postleitzahl von einem Bild identifizieren.
- Risikofaktoren für Krebs aufgrund klinischen und demografischen Daten abschätzen.

Man unterscheidet zwischen mehrere Arten von Machine Learning. Die Hauptarten sind supervised (beaufsichtigt) und unsupervised (unbeaufsichtigt) learning. Das beaufsichtigte Lernen verwenden die Eingabewerte *X* und die vorklassifizierten Daten *Y*. Das Zielist, solche Funktion *f()* zu finden, damit *f(x) = y*. Dann kann man diese Funktion auf nicht vorklassifizierten Daten anwenden, damit Klassifikationen herauskommen.

Beim unbeaufsichtigten Lernen haben wir nur Eingabedaten *X*. Das Ziel ist Stukturen oder Muster innerhalb der Daten zu finden, mit deren Hilfe man mehr Information über die Daten erfahren kann. Diese Art von Lernen ist 'unbeaufsichtigt' genannt, weil es keine 'richtige' Antwort gibt.

Noch zwei wichtige Begriffe sind *Klassifizierung* und *Regression*. Klassifizierung ist wenn unsere Ausgabendaten *Y* keine metrische Darstellung haben. Zum Beispiel aufgrund Gewicht und Größe die Art von Hühner zu bestimmet. Regression wäre dann zum Beispiel, wenn wir den Preis eines Haus aufgrund ihre Lage, Größe und Zimmeranzahl bestimmen.

## Datensatz holen

Ich habe die Daten vom [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Iris) geholt. Da kann man viele verschiedene Datensätze finden, die alle kostenlos sind. 

Der Datensatz den ich gewählt habe ist sehr bekannt in der Machine-Learning-Literatur. Er bezieht sich auf drei Arten der Iris Blume. Gegeben sind Sepalumlänge, Sepalumbreite, Petalumlänge, Petalumbreite und Klasse von 150 Individuen. Wir wollen jetzt einen Algorithmus trainieren, Individuen aufgrund den Längen und Breiten zu klassifizieren. Wir verwenden dafür den kNN Algorithmus.

*insert image for petal/sepal here*

## Der kNN Algorithmus

KNN steht für *k nearest neighbours (k nächste Nachbarn)*. Das ist ein robuster und beaufsichtigter Algorithmus, der sowohl für Klassifizierung als auch für Regression benutzt werden kann. Zusätzlich wird kein Modell erstellt, sondert werden alle Trainingsdaten bei jeder Klassifizierung geladen.

Das bedeutet, dass kNN braucht keine Zeit für trainieren, dafür gibt entstehen hohe Hauptspeicherkosten bei der Klassifizierungsphase, da wir oft einen großen Datensatz laden müssen. Außerdem ist die Berechnung auch teuer, weil wir für jede Klassifizierung über den ganzen Datensatz iterieren müssen.

## Wie funktioniert kNN?

KNN plottet alle Trainingsdaten in einem N-dimensionalen Vektorraum, wobei *n* die Dimension der Eingabevektor ist. In unserem Fall ist *n=4*, weil unsere Eingabewerte Sepalumlänge und -breite und Petalumlänge und -breite sind. 

Danach für jedes Element das wir klassifizieren wollen, berechnen wir den Abstand zwischen das Element und jeden Mitglied der Trainingdaten. Am meistens verwenden man den euklidischen Abstand, aber es funktioniert auch mit anderen Abstandsmetriken. 

*formula for euclidian distance here*

Nachdem wir alle Abstände haben, nehmen wir *K* Elemente, mit dem kleinen Abstand. Abhängig davon, ob wir ein Klassifikationsproblem oder ein Regressionsproblem haben, nehmen wir entweder die am meisten vorkommende Klasse als Ergebnis bezugsweise den Durchschnitt des *Y* Parameters der Trainingsdaten.

Wie wählt man ein *K* denn? Das ist eine gute Frage. 


