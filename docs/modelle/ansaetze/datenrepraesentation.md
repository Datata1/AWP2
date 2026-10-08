# Datenrepräsentation

Quer zu allen Ansätzen: Wie sieht das Modell das Spektrum?

| Sicht | Idee | Typische Klassifikatoren |
| --- | --- | --- |
| Tabellarisch | Jedes Band ist ein eigenes Feature | Random Forest, Gradient Boosting, SVM, MLP |
| Sequenziell | Das Spektrum ist eine geordnete Folge entlang der Wellenlänge (keine Zeitreihe) | 1D-CNN, RNN/LSTM/GRU, Transformer |

## Entscheidung für M2

**Zuerst tabellarisch arbeiten.** Tabellarische Modelle passen zur vorhandenen sklearn-Pipeline,
erlauben einen fairen Vergleich auf dem festen Split und sind bei rund 3.900 Trainingszeilen
deutlich weniger überanpassungsanfällig als tiefe Netze. Welche tabellarische Konfiguration die
Referenz bildet, entscheidet jede Ansatzseite anhand ihrer Experimente.

Sequenziell ist kein anderer Datensatz: Dasselbe Spektrum wird als Folge der nach Wellenlänge
sortierten Bänder betrachtet. Die Bandachse beschreibt jedoch keine Zeit; deshalb sind 1D-CNNs
mit lokalen Filtern der erste sequenzielle Kandidat. RNNs und Transformer folgen nur bei einem
klaren Nutzen der CNN oder einer neuen Hypothese.

| Frage | Tabellarisch | Sequenziell |
| --- | --- | --- |
| Eingabe | Eine Zeile mit Band-, AEZ- und Month-Features | Bandfolge der Form `(Beobachtungen, Bänder, 1)`; AEZ/Month getrennt als Kontextfeatures |
| Sinnvoller erster Kandidat | RBF-SVM, Extra Trees oder HistGradientBoosting | Kleine 1D-CNN mit wenigen Filtern und frühem Stopp |
| Skalierung | Nur für SVM, lineare Modelle und MLP | Immer für die Bandfolge, nur auf dem jeweiligen Trainings-Fold gelernt |
| Stärke | Robuste Referenz bei wenig Daten; Kontext leicht einbeziehbar | Kann lokale Form, Kanten und Absorptionsmuster lernen |
| Hauptrisiko | Ignoriert die Nachbarschaft der Wellenlängen explizit | Viele Parameter bei kleinen und unausgewogenen Klassen |

## Reihenfolge der Untersuchung

1. Je Ansatz eine tabellarische Referenz mit dessen Zielrepräsentation reproduzieren und nur
    kontrollierte Erweiterungen prüfen: Klassifikator, Kontext-Ablation, Vegetationsindizes oder
    eine begründete spektrale Transformation.
2. Erst wenn die tabellarische Vergleichsbasis dokumentiert ist, eine kleine 1D-CNN als
    sklearn-kompatiblen Wrapper bauen. Sie bekommt genau dieselben Trainings-Folds,
    Zielrepräsentation, Metriken und den gleichen Validierungs-Holdout wie die Tabellenmodelle.
3. Die CNN mit einer kleinen Suche für Filterzahl, Kernelgröße, Regularisierung und Epochen
    per `tune()` vergleichen. Der Validierungsscore entscheidet nicht über weitere Epochen oder
    Architekturversuche.
4. Nur bei einem stabilen CV-Gewinn und nachvollziehbarer Validierungsverbesserung komplexere
    Sequenzmodelle prüfen. Ohne diesen Befund bleiben RNN, LSTM, GRU und Transformer außerhalb
    des M2-Umfangs.

## Bandlücken und Kontext bei Sequenzmodellen

Die Daten enthalten spektrale Lücken, weil vollständig leere Bänder entfernt werden. Eine
1D-CNN darf daher nicht so behandelt werden, als ob alle benachbarten Werte den gleichen
Wellenlängenabstand hätten. Der erste Prototyp soll die erhaltenen Bandpositionen explizit
dokumentieren und die Lücken nur innerhalb zusammenhängender Wellenlängenbereiche falten,
oder die normierte Wellenlänge als zusätzlichen Kanal erhalten. Ein Auffüllen erfundener
vollständig leerer Bänder ist keine Standardoption.

AEZ und Month sind keine Spektralwerte. Sie werden entweder nach dem CNN-Teil in einem kleinen
Kontextzweig mit den gelernten Spektralmerkmalen verbunden oder in einer klaren Ablation ganz
weggelassen. Das Zusammenhängen als künstliche weitere Bandposition ist nicht zulässig.

## Abbruchkriterien

Ein sequenzieller Ansatz wird verworfen oder zurückgestellt, wenn er die tabellarische Referenz
des jeweiligen Ansatzes in der CV nicht übertrifft, wenn der Gewinn nur auf der Validierung
erscheint oder wenn seltene Klassen deutlich schlechter werden. Dann bleibt die Erkenntnis
trotzdem wertvoll: Die tabellarische Darstellung bildet die für diese Daten verfügbare
Information ausreichend ab.

!!! todo "Leitfragen"
    - Was probieren wir, was bringt die sequenzielle Sicht gegenüber der tabellarischen?
    - Wie gehen sequenzielle Modelle mit den Lücken der entfernten Bänder um?
