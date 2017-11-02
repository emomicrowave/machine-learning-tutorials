## Notes and ideas

- explain what is a support vector machine
- simple use cases
- tricks and graphs about overlapping data
- mention non-linear SVN
- implementation with scikit learn
- pros and cons
- (?) Mapping to infinite dimensions
- (?) SVN Parameter in scikit-learn

# Support Vector Machine (SVN)
Support Vector Machine ist ein überwachtes Machine Learning Verfahren, überwiegend für binäre Klassifizierung benutzt.
Die Trainingsdaten werden im n-dimensionalen Raum geplottet und der Algorithmus versucht eine solche Grenze zwischen
den beiden Klassen zu finden, so dass ein möglichst breiter Bereich frei von Stichproben um die Grenze herum ensteht.

Eigentlich geht es hier um eine Hyperebene. Das ist ein Unterraum, dessen Dimension um 1 kleiner ist als seine Umgebung. 
Zum Beispiel im dreidimensionalen Raum, wäre eine Hyperebene eine zweidimensionale Ebene. Und im zweidimensionalen Raum wäre
eine Hyperebene einfach eine gerade Linie. 

Daraus folgt aber, dass die Daten linear trennbar sind, was für die meisten reellen Fälle nicht der Fall ist. 
Deswegen kann man den sogenannten *Kernel-Trick* verwenden. Die Idee dahinter ist das Vektorraum in einem höherdimensionalen 
Raum zu überführen, wobei die Objekte schon linear trennbar sind und da eine Hyperebene zu definieren. 
Bei der Rücktransformation, wird diese Hyperebene nichtlinear und oft auch nicht zusammenhängend.

Probleme sind der hohe Rechenaufwand bei der Dimensionsteigerung und die oft nicht intuitive und unbrauchbare Form der
Hyperebene nach der Dimensionsreduktion. *kernel functions here*

*kernel functions - expanding the dataset and there are now more obvious boundaries between the classes:*

## Beispiele

## Implementierung mit Python

## Ressourcen
- Logistische Regression vs. Decision Trees vs. Support Vector Machines - [Link](https://www.edvancer.in/logistic-regression-vs-decision-trees-vs-svm-part1/)
