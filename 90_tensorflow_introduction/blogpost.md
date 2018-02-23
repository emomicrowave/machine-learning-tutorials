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
  - ??

- **essential datasets**
  - MNIST
  - CIFAR10 / 100
  - (?) Zalando

- **links to the TF official website as 'get started'**
  - (?) a neural network tutorial is coming

# Deep Learning mit TensorFlow und mögliche Anwendungsfälle
*Deep Learning* ist kein neues Konzept im maschinellen Lernen, aber es ist in der letzten Jahren sehr populär geworden. Eigentlich ist *Deep Learning* nur ein schicker Name für künstliche mehrschichtige neuronale Netze. Die Idee dahinter war, einen selbstlernenden Algorithmus zu entwickeln, der das menschliche Gehirn nachahnt. Neuronale Netze bestehen aus Schichten (Neuronen), die mit benachbarten Schichten verbunden sind. Je mehr Schichten es gibt, desto "tiefer" das Netz.

Ein Merkmal von *Deep Learning* Modellen ist die große Menge an Daten, die sie für effektives Training benötigen. Dies führt zusammen mit den vielen Schichten und Neuronen innerhalb des Netzwerkes zu der Notwendigkeit einer hohen Rechenleistung. Um das zu erreichen sollen Deep Learning Modelle auf Servern mit mehreren Grafikkarten laufen können, oder sogar auf verteilten Systemen und Cloud Computing Diensten wie z.B Google Cloud, AWS oder Azure. Damit hilft auch TensorFlow.

*TensorFlow* ist eine Open Source Bibliothek für Machine Learning, die von Google erstellt worden ist. Am Anfang war die Idee diese Framework für interne Zwecke zu verwenden. Trotzdem wurde die unter eine Open Source Lizenz veröffentlichen, mit dem Ziel der Deep Learning und Machine Learning Forschung. TensorFlow ist nicht nur ideal für Forschung, sondern auch für echte Produkte, da er schnell, übertragbar und einsatzbereit ist.

Dies ist machbar, weil TensorFlow es ermöglicht, ein Modell auf einer oder mehreren GPUs zu beitreiben oder einfach in verteilte Systeme zu integrieren. Nicht nur das, sondern bietet er auch verschiedene APIs damit man einfach ein Machine Learning Modell erstellen, trainieren und bereitstellen kann.

## Wofür nutzen Unternehmen TensorFlow
TensorFlow hat viele Stärke wie z.B Bilderkennung. Firmen wie Amazon und Netflix verwenden es um eine Datenbank zu erstellen mit welchen Objekten sich in verschiedenen Szenen aus Filmen oder Serien befinden. Zusätzlich wird auch Gesichtserkennung angewendet, um eine Liste von Schauspielern in der aktuellen Szene. 

Eine andere Anwendung von Bilderkennung mit TensorFlow ist in der Landwirschaft. Indem ein ehemaliger Designer der japanischen Automobilindustrie seiner Familie auf der Gurkenfarm half, entdeckte er, dass das Sortieren der Gurken eine sehr anstrengedne Aufgabe ist, die seine Elter manuell ausführen. Als Setup benutzte er einen Raspberry Pi mit einer Kamera und einem kleinen neuronalen Netzwerk, das prüft ob das Bild eine Gurke ist. Wenn dies der Fall ist, wird das Bild zur detaillierteren Klassifizierung an ein größeres neuronales Netzwerk auf einem Linux-Server gesendet. 


