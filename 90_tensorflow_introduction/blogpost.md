### Ideas and stuff about the blog post
- **What is TensorFlow**
  - what is deep learning
  - what is TensorFlow - not build from scratch

- **business use cases**
  - cucumber sorting
  - amazon: items you may like
  - NLP to see what the email is about
  - maybe something about movie object + face detection (you only look once / object localization)
  - how are these achieveable? (deep nets on clusters)

- **other cool projects**
  - neural style
  - pretrained models

- **essential datasets**
  - MNIST
  - CIFAR10 / 100
  - ImageNet
  - (?) Zalando

- **links to the TF official website as 'get started'**
  - (?) a neural network tutorial is coming

# Deep Learning mit TensorFlow und mögliche Anwendungsfälle
*Deep Learning* ist kein neues Konzept im maschinellen Lernen, aber es ist in der letzten Jahren sehr populär geworden. Eigentlich ist Deep Learning nur ein schicker Name für künstliche mehrschichtige neuronale Netze. Die Idee dahinter war, einen selbstlernenden Algorithmus zu entwickeln, der das menschliche Gehirn nachahnt. Neuronale Netze bestehen aus Schichten (Neuronen), die mit benachbarten Schichten verbunden sind. Je mehr Schichten es gibt, desto "tiefer" das Netz.

Ein Merkmal von Deep Learning Modellen ist die große Menge an Daten, die sie für effektives Training benötigen. Dies führt zusammen mit den vielen Schichten und Neuronen innerhalb des Netzwerkes zu der Notwendigkeit einer hohen Rechenleistung. Um das zu erreichen sollen Deep Learning Modelle auf Servern mit mehreren Grafikkarten laufen können, oder sogar auf verteilten Systemen und Cloud Computing Diensten wie z.B Google Cloud, AWS oder Azure. Damit hilft auch TensorFlow.

*TensorFlow* ist eine Open Source Bibliothek für Machine Learning, die von Google erstellt worden ist. Am Anfang war die Idee diese Framework für interne Zwecke zu verwenden. Trotzdem wurde die unter eine Open Source Lizenz veröffentlichen, mit dem Ziel der Deep Learning und Machine Learning Forschung. TensorFlow ist nicht nur ideal für Forschung, sondern auch für echte Produkte, da er schnell, übertragbar und einsatzbereit ist.

Dies ist machbar, weil TensorFlow es ermöglicht, ein Modell auf einer oder mehreren GPUs zu beitreiben oder einfach in verteilte Systeme zu integrieren. Nicht nur das, sondern bietet er auch verschiedene APIs damit man einfach ein Machine Learning Modell erstellen, trainieren und bereitstellen kann.

## Wofür nutzen Unternehmen TensorFlow
TensorFlow hat viele Stärke wie z.B Bilderkennung. Firmen wie Amazon und Netflix verwenden es um eine Datenbank zu erstellen mit welchen Objekten sich in verschiedenen Szenen aus Filmen oder Serien befinden. Zusätzlich wird auch Gesichtserkennung angewendet, um eine Liste von Schauspielern in der aktuellen Szene.

Eine andere Anwendung von Bilderkennung mit TensorFlow ist in der Landwirschaft. Indem ein ehemaliger Designer der japanischen Automobilindustrie seiner Familie auf der Gurkenfarm half, entdeckte er, dass das Sortieren der Gurken eine sehr anstrengedne Aufgabe ist, die seine Elter manuell ausführen. Als Setup benutzte er einen Raspberry Pi mit einer Kamera und einem kleinen neuronalen Netzwerk, das prüft ob das Bild eine Gurke ist. Wenn dies der Fall ist, wird das Bild zur detaillierteren Klassifizierung an ein größeres neuronales Netzwerk auf einem Linux-Server gesendet. Der Designer musste zuerst 7000 Bilder von sortierten Gurken manuell aufnehmen. Dann dauerte das Training des Modells 2 oder 3 Tage, weil es auf einem normalen Windows-PC lief. Obwohl das Setup relativ einfach ist, hat es 70% der Gurken korrekt klassifiziert und zeigt ein großes Verbesserungspotential.

Hier ist ein weiterer Anwendungsfall von TensorFlow: Firmen erhalten jeden Tag zu viele E-Mails. Bis vor kurzem wurden sie meist in der Reihenfolge ihrer Ankunf verarbeitet. Ein Problem wurde in Notfällen entstehen, wenn das Unternehmen zu viele E-Mails erhält, um sie effektiv zu verarbeiten. Eine Möglichkeit, dieses Problem zu lösen, besteht darin, *Natural Language Processing* zu verwenden, um die Stimmung und das Thema eingehender E-Mails zu verstehen und ihnen automatisch Priorität zuzuweisen.

## Interessante Projekte
### Neural Style
Eine interessante Anwendung von TensorFlow sieht man im [*Neural Style* Projekt](https://github.com/cysmith/neural-style-tf) auf Github. Der Algorithmus synthesiert einen Pastiche: ein künstlerisches Werk, das offen das Werk eines vorangegangenen Künstlers imitiert. In diesem Fall wird der Inhalt eines Bildes mit dem Stil eines anderen Bildes kombiniert. Unten sieht man eine Reihe von Häusern im Tübingen mit Blick auf den Neckar, kombiniert mit den Kunststilen verschiedener Gemälde.

![neural style example](images/neural_style.png)

*arXiv - [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) - (2015) Gatys, Ecker, Bethge*

### Transfer Learning
Was TensorFlow auch bietet, sind vortrainierte Modelle für Objekterkennung in Bildern, die sich perfekt für *Transfer Learning* eignen. Transfer Learning besteht darin, die bereits vorhandenen ersten Schichten der Modelle und die nützlichen Abstraktionen drin zu verwenden und so nur die obersten Schichten von Grund auf zu trainieren. Die Verwendung dieser Methode führt zu viel schnelleren Trainingszeiten und man benötigt nicht so große Datensätze, um eine gute Genauigkeit zu erreichen. Ein Beispiel wäre das [ Umtrainieren des *Inception*-Modells zur Klassifizierung von Blumen. ](https://www.tensorflow.org/tutorials/image_retraining)

Andere Modelle kann man hier finden: [Link](https://github.com/tensorflow/models)

### Sonstige
Die Github-Repository [*Awesome TensorFlow*](https://github.com/jtoy/awesome-tensorflow) enthält zahlreiche interessante Projekte sowie Tutorials, Videos und Blogbeiträge. 
