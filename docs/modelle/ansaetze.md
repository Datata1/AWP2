# Ansätze

Die Aufgabe lässt offen, wie wir Kultur und Stadium modellieren; die Wahl muss **begründet**
und mit Alternativen verglichen werden (siehe [Aufgabenstellung](../projekt/index.md)).
Hier steht je Ansatz, was die Idee ist, was dafür und dagegen spricht und was die Experimente
ergeben haben. Begriffe: [ML-Aufgabe](../domaene/ml-aufgabe.md#problemformulierung).

## Überblick

| Ansatz | Status | Bestes Experiment | Entscheidung |
| --- | --- | --- | --- |
| Baseline | geplant | | |
| Getrennte Modelle (Crop, Stage) | offen | | |
| Kombinierte Klasse (Crop_Stage) | offen | | |
| Hierarchisch (erst Crop, dann Stage) | offen | | |
| Multi-Task / Multi-Output | offen | | |

Status: *offen* → *geplant* → *in Arbeit* → *verglichen* → *gewählt* / *verworfen*.

## Baseline

!!! todo "Leitfragen"
    - Welche Referenz schlagen wir mindestens (Dummy-Classifier, einfaches Modell auf den
      gültigen Bändern)? Welche Scores erreicht sie?

## Getrennte Modelle

!!! todo "Leitfragen"
    - Idee: je ein Modell für Crop und Stage. Vorteil: einfach. Nachteil: kann unmögliche
      Kombinationen vorhersagen, Stage ignoriert die Kultur.

## Kombinierte Klasse

Baustein: `awp2.models.combined.CombinedLabelClassifier` (#54).

**Was es modelliert:** Kultur und Stadium werden zu *einer* Klasse `Crop|Stage` verbunden –
23 Klassen im Datensatz. Ein beliebiger Klassifikator lernt damit direkt die gemeinsame
Verteilung P(Crop, Stage).

**Erste Einschätzung (noch zu belegen):**

| Vorteil | Nachteil |
| --- | --- |
| Sagt nur Paare vorher, die es gibt | Nutzt nicht, dass das Stadium von der Kultur abhängt: P(Crop, Stage) = P(Crop) · P(Stage \| Crop) wird als 23 unabhängige Klassen gelernt |
| Funktioniert mit jedem Klassifikator | Kein geteiltes Wissen zwischen z. B. Mais/Late und Soja/Late |
| Ein Modell, einfach zu vergleichen | Wenige Daten je Klasse (Baumwolle/Harvest: 11 Zeilen) |
| | Jeder Kultur-Fehler ist automatisch auch ein Stadium-Fehler |

!!! todo "Recherche (#54, Grundlagen in #11)"
    - Wird der Ansatz in der Literatur für hierarchische Labels genutzt, und mit welchem Ergebnis?
    - Wie verhält er sich gegenüber hierarchischer Klassifikation bei so seltenen Klassen?
    - Quellen in [Quellen](../domaene/quellen.md) eintragen.

## Hierarchisch

!!! todo "Leitfragen"
    - Idee: erst Crop, dann ein Stage-Modell je Kultur. Wie pflanzen sich Crop-Fehler fort?
      Reichen die Daten pro Kultur (z. B. Reis: 93 Zeilen)?

## Multi-Task / Multi-Output

!!! todo "Leitfragen"
    - Idee: ein Modell mit zwei Ausgaben (z. B. neuronales Netz mit zwei Köpfen). Lohnt der
      Aufwand gegenüber den anderen Ansätzen?

## Datenrepräsentation

!!! todo "Leitfragen"
    - Tabellarisch (Bänder als Features: Random Forest, Gradient Boosting, SVM) vs. sequenziell
      (Spektrum als Sequenz: 1D-CNN, …) – was probieren wir, was bringt es?
