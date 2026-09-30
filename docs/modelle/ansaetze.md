# Ansätze

Die Aufgabe lässt offen, wie wir Kultur und Stadium modellieren; die Wahl muss **begründet**
und mit Alternativen verglichen werden (siehe [Aufgabenstellung](../projekt/index.md)).
Hier steht je Ansatz, was die Idee ist, was dafür und dagegen spricht und was die Experimente
ergeben haben. Begriffe: [ML-Aufgabe](../domaene/ml-aufgabe.md#problemformulierung).

## Überblick

| Ansatz | Status | Bestes Experiment | Entscheidung |
| --- | --- | --- | --- |
| Baseline | verglichen | `rf_baseline` (BAcc kombiniert 0.743) | Referenz für alle weiteren Ansätze |
| Getrennte Modelle (Crop, Stage) | offen | | |
| Kombinierte Klasse (Crop_Stage) | offen | | |
| Hierarchisch (erst Crop, dann Stage) | offen | | |
| Multi-Task / Multi-Output | offen | | |

Status: *offen* → *geplant* → *in Arbeit* → *verglichen* → *gewählt* / *verworfen*.

## Baseline

Referenzwerte für M1 (#27, Notebook `notebooks/01_jd_baseline.ipynb`) – bewusst **kein**
begründeter Ansatz, sondern die Messlatte für alle weiteren.

| Modell | BAcc Crop | BAcc Stage | BAcc kombiniert | Samples-F1 |
| --- | --- | --- | --- | --- |
| Dummy (immer die häufigste Klasse) | 0.200 | 0.167 | 0.044 | 0.326 |
| Random Forest (Crop und Stage gemeinsam) | 0.890 | 0.870 | 0.743 | 0.878 |

**Random Forest:** 300 Bäume, `class_weight="balanced"`, Vorverarbeitung mit AEZ/Month.
Per CV gesucht (8 Kandidaten): beste Einstellung `max_depth=20`, `max_features=0.3`,
`min_samples_leaf=1` – CV-Score 0.750 ± 0.019, Validierung 0.743.

- CV- und Validierungs-Score liegen dicht beieinander → keine Überanpassung an den Holdout.
- `max_features=0.3` schlägt `sqrt` deutlich (0.75 vs. 0.66–0.70): Pro Split mehr Bänder zu
  sehen hilft – die Information steckt in vielen, stark korrelierten Bändern.
- Kultur: häufigste Verwechslung **Mais ↔ Soja** (45 bzw. 31 Fälle) – beides Sommerkulturen mit
  ähnlicher Saison; außerdem Winterweizen → Baumwolle.
- Stadium: vor allem **benachbarte Stadien** (Emerge_VEarly → Early_Mid, Critical →
  Mature_Senesc); schwächstes Stadium Mature_Senesc (Recall 0.79).
- 0,3 % der Vorhersagen sind unmögliche Kultur/Stadium-Paare – der Random Forest kennt die
  Abhängigkeit nicht. Argument für hierarchische oder kombinierte Ansätze (M2).

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
