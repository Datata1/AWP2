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

!!! todo "Leitfragen"
    - Idee: Crop und Stage zu einer Klasse (`corn_Late`, …). Wie viele Klassen entstehen, wie
      selten sind die kleinsten? Nur gültige Kombinationen möglich.

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
