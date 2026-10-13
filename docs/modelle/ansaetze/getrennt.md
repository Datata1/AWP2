# Getrennte Modelle

## Idee

Je ein Modell für Crop und für Stage wird auf allen Trainingszeilen unabhängig trainiert. Die
beiden Vorhersagen werden anschließend zu einem Crop/Stage-Paar zusammengefügt. Damit modelliert
der Ansatz die gemeinsame Verteilung als $P(\text{Crop}) \cdot P(\text{Stage})$ und nutzt weder
die Crop-Information für die Stadiumsvorhersage noch gemeinsame Modellparameter.

Dies ist weder der [kombinierte Ansatz](kombiniert.md), der ein gemeinsames Label lernt, noch der
[hierarchische Ansatz](hierarchisch.md), dessen Stadium-Modelle nur auf Zeilen einer Kultur
trainieren. Er dient als Kontrolle dafür, ob die explizite Kopplung von Crop und Stage den
zusätzlichen Modellaufwand rechtfertigt.

## Vor- und Nachteile

| Vorteil | Nachteil |
| --- | --- |
| Zwei klar getrennte, einfache Aufgaben; Crop und Stage können je eigene Hyperparameter erhalten. | Das Stage-Modell ignoriert, dass nicht jede Kultur jedes Stadium hat. |
| Keine Fehlerfortpflanzung durch eine vorgeschaltete Crop-Entscheidung. | Es können Crop/Stage-Paare vorhergesagt werden, die im Training nicht vorkommen. |
| Direkter Vergleich dazu, ob die Abhängigkeit der Targets überhaupt hilft. | Das globale Stage-Modell muss Stadien über Kulturen hinweg trennen und kann dadurch schwerer lernen. |

Die erste Variante korrigiert ungültige Paare bewusst nicht nachträglich. Ihr Anteil ist eine
eigene Metrik von `evaluate()` und macht den Nachteil des Ansatzes sichtbar. Eine spätere
Paarbeschränkung über Modellwahrscheinlichkeiten wäre eine getrennte Hypothese, weil sie die
Unabhängigkeit der beiden Vorhersagen nachträglich aufhebt.

## Klassifikatoren

### RBF-SVM -> RBF-SVM

**Hypothese:** Zwei global trainierte, ausgeglichene RBF-SVMs erreichen ohne bedingtes Routing
eine mindestens so hohe kombinierte BAcc wie die hierarchische RBF-SVM. Falls nicht, zeigt der
Abstand den Nutzen, die Stage-Vorhersage von der Kultur abhängig zu machen.

#### Erster Arbeitsplan

1. `IndependentClassifier` in `src/awp2/models/independent.py` implementieren. Er klont
    `crop_estimator` und `stage_estimator`, trainiert sie auf `y["Crop"]` bzw. `y["Stage"]` und
    gibt eine DataFrame-Vorhersage mit beiden Zielspalten zurück. Die Parameter bleiben als
    `crop_estimator__...` und `stage_estimator__...` für `tune()` erreichbar.
2. `make_independent_svm()` exportieren: zwei `SVC(class_weight="balanced")`. Der erste Versuch
    nutzt die Vorverarbeitung der kombinierten SVM-Referenz: skalierte Spektren, AEZ und Month.
    Nicht im Notebook skalieren oder Labels verändern.
3. Einen Unit-Test schreiben: Beide Estimatoren werden gefittet, die Vorhersage hat exakt die
    Spalten `Crop` und `Stage`, und unbekannte Paare bleiben erlaubt. Das ist der günstige Test,
    ob der Adapter mit `tune()` und `run()` kompatibel sein kann.
4. Im Notebook `18_<initials>_independent-svm.ipynb` die getrennten SVMs per `tune()` nur auf den
    fünf Trainingsfolds suchen. Anfangsraster: `C` in {30, 100, 300} und `gamma` in
    {0.001, 0.003, 0.01} je Modellstufe; das sind 81 Kombinationen.
5. Ausschließlich den CV-Gewinner einmal mit `run()` auf dem gemeinsamen Validierungssplit
    bewerten. Dokumentieren: BAcc für Crop, Stage und kombiniert, Macro-F1, Samples-F1 sowie den
    Anteil ungültiger Paare.
6. Gegen `combined_svm_expanded` und `hierarchical_svm` vergleichen: Confusion Matrices,
    Klassenrecall und `error_breakdown()` zeigen, ob die fehlende Fehlerfortpflanzung oder die
    fehlende Bedingung auf die Kultur überwiegt. Keine Paarbeschränkung nach dem Validierungsscore
    ergänzen.
7. Tuning- und Bewertungslauf in MLflow verknüpfen, eine Zeile im
    [Experiment-Log](../experimente.md) ergänzen und diese Seite mit Ergebnis, MLflow-Run und
    Entscheidung aktualisieren.

**Entscheidungsregel:** Der Ansatz wird nur weiter verfolgt, wenn seine CV-BAcc kombiniert die
hierarchische Referenz von 0.839 belastbar erreicht oder übertrifft. Bei ähnlicher BAcc ist der
Anteil ungültiger Paare ausschlaggebend; die kombinierte SVM bleibt als Referenz, wenn sie ohne
Paar-Korrektur klar besser ist.

## Fazit

**Geplant.** Der unabhängige RBF-SVM-Vergleich schließt die noch offene Modellierungsstrategie.
Er testet gezielt die Rolle der Target-Abhängigkeit, nicht eine weitere allgemeine
Hyperparameter-Variation.
