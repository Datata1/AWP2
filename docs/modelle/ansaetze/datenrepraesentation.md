# Datenrepräsentation

Quer zu allen Ansätzen: Wie sieht das Modell das Spektrum?

| Sicht | Idee | Typische Klassifikatoren |
| --- | --- | --- |
| Tabellarisch | Jedes Band ist ein eigenes Feature | Random Forest, Gradient Boosting, SVM, MLP |
| Sequenziell | Das Spektrum ist eine geordnete Folge entlang der Wellenlänge (keine Zeitreihe) | 1D-CNN, RNN/LSTM/GRU, Transformer |

!!! todo "Leitfragen"
    - Was probieren wir, was bringt die sequenzielle Sicht gegenüber der tabellarischen?
    - Wie gehen sequenzielle Modelle mit den Lücken der entfernten Bänder um?
