# Klassifikatoren

Diese Seite sammelt mögliche Klassifikatoren für die Crop/Stage-Aufgabe. Sie ist ein
**Kandidatenkatalog**, kein Experiment-Log: Ein Eintrag hier bedeutet nicht, dass das Modell
bereits trainiert oder bewertet wurde. Ergebnisse stehen erst nach `tune()` und `run()` in
[Experimente](experimente.md) und auf der zugehörigen [Ansatzseite](ansaetze/index.md).

## Einordnung

Ein Klassifikator beantwortet, *welcher Algorithmus* aus den Merkmalen lernt. Davon getrennt
steht der [Modellierungsansatz](ansaetze/index.md): ob Crop und Stage getrennt, kombiniert,
hierarchisch oder als Multi-Task-Aufgabe vorhergesagt werden. Viele tabellarische
Klassifikatoren lassen sich mit mehreren Ansätzen kombinieren.

## Tabellarische Klassifikatoren

Diese Modelle erhalten die 198 Spektralbänder sowie optional AEZ und Month als eine
Merkmalszeile. Sie sind der erste Vergleich, weil sie mit der vorhandenen sklearn-Pipeline
direkt nutzbar und bei rund 3.900 Trainingszeilen gut kontrollierbar sind.

| Familie | Kandidaten | Voraussetzungen | Priorität |
| --- | --- | --- | --- |
| Untergrenze | DummyClassifier | keine | Referenz |
| Baum-Ensembles | Random Forest, Extra Trees | Klassen- oder Stichprobengewichte | hoch |
| Gradient Boosting | HistGradientBoosting | Stichprobengewichte; fehlende Werte prüfen | hoch |
| Kernelmethoden | SVM mit RBF- oder linearem Kernel | Skalierung | hoch |
| Lineare Modelle | Logistische Regression, lineare SVM, LDA | Skalierung für logistische/lineare SVM | mittel |
| Neuronales Tabular-Modell | MLP | Skalierung, Regularisierung, kleine Suche | mittel |
| Distanzbasiert | k-Nearest Neighbors | Skalierung; Nachbarn abstimmen | niedrig |
| Probabilistisch | Gaussian Naive Bayes | Verteilungsannahmen kritisch prüfen | niedrig |

Random Forest ist bereits als [Baseline](ansaetze/baseline.md) und für die
[kombinierte Klasse](ansaetze/kombiniert.md) getestet. Zusätzliche Bibliotheken für XGBoost,
LightGBM oder CatBoost werden nur aufgenommen, wenn eine begründete Hypothese ihren Zusatznutzen
gegenüber den vorhandenen sklearn-Modellen rechtfertigt.

## Sequenzklassifikatoren

Hier bleibt die Bandreihenfolge entlang der Wellenlänge erhalten. Die Achse beschreibt keine
Zeitreihe, sondern die spektrale Nachbarschaft der Bänder. Diese Modelle sind spätere Vergleiche,
weil sie zusätzliche Implementierung und bei der kleinen Stichprobe stärkere Regularisierung
benötigen.

| Familie | Kandidaten | Erwarteter Nutzen | Risiko |
| --- | --- | --- | --- |
| Lokale Faltung | 1D-CNN | Muster wie die Red Edge direkt lernen | Überanpassung, Umgang mit Bandlücken |
| Rekurrent | RNN, GRU, LSTM | Abhängigkeiten über mehrere Bänder | Wellenlänge ist keine Zeitreihe; hoher Aufwand |
| Attention | Transformer | weit entfernte Bänder verknüpfen | wahrscheinlich datenhungrig |
| Hybrid | 1D-CNN mit Attention | lokale und globale Muster verbinden | höchste Komplexität |

## Priorisierter Vergleichsplan

Die Liste ist kein Auftrag, jede Kombination zu testen. Ein sinnvoller Vergleich deckt
unterschiedliche Hypothesen ab und bleibt auf dem gemeinsamen Split nachvollziehbar.

1. Kombinierte Klasse mit Random Forest: abgeschlossen, aktuelle Referenz für BAcc kombiniert.
2. Hierarchischer Ansatz mit Random Forest: prüft, ob kulturabhängige Stadien helfen.
3. Kombinierte Klasse mit SVM: testet eine nichtlineare Entscheidungsgrenze mit Skalierung.
4. Kombinierte Klasse mit Extra Trees oder HistGradientBoosting: testet alternative
   Baum-Ensembles.
5. Getrennte Random-Forest-Modelle: misst den Nutzen der gemeinsamen Labelstruktur.
6. Multi-Task-MLP: prüft geteilte Repräsentationen bei tabellarischen Merkmalen.
7. 1D-CNN: vergleicht die tabellarische mit der sequenziellen Spektralsicht.

Jeder Kandidat durchläuft denselben Ablauf: Hyperparameter mit `tune()` nur auf dem
Train-Anteil auswählen, mit `run()` genau einmal auf der Validierung bewerten und anschließend
gegen die bisherige Referenz vergleichen. Der Validierungsscore entscheidet nicht über weitere
Hyperparameter-Versuche.