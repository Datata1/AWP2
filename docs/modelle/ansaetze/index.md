# Ansätze

Die Aufgabe lässt offen, wie wir Kultur und Stadium modellieren; die Wahl muss **begründet**
und mit Alternativen verglichen werden (siehe [Aufgabenstellung](../../projekt/index.md)).
Begriffe: [ML-Aufgabe](../../domaene/ml-aufgabe.md#problemformulierung).

- Ein **Ansatz** legt fest, *wie* Crop und Stage gemeinsam vorhergesagt werden (getrennt,
  kombiniert, hierarchisch …) – eine Seite je Ansatz.
- Ein **Klassifikator** ist das Modell darin (Random Forest, SVM, 1D-CNN …) – ein Unterkapitel
  je Klassifikator auf der Seite des Ansatzes. Zweistufige Ansätze (getrennt, hierarchisch)
  können je Stufe einen anderen nutzen; dann heißt das Unterkapitel nach der Kombination, z. B.
  „Random Forest → SVM“ (Kultur → Stadium).

## Überblick

| Ansatz | Status | Klassifikatoren | Bestes Experiment | Entscheidung |
| --- | --- | --- | --- | --- |
| [Baseline](baseline.md) | verglichen | Dummy, Random Forest | `rf_baseline` (BAcc kombiniert 0.743) | Referenz für alle weiteren Ansätze |
| [Getrennte Modelle](getrennt.md) | offen | – | | |
| [Multi-Task](multi-task.md) | verglichen | Tabellarisches MLP, LSTM | `multi_task_mlp_pca20` (BAcc kombiniert 0.765) | LSTM mit Embeddings erreicht 0.417 und wird nicht weiter verfolgt; die anderen Kontextvarianten bleiben unbewertet |
| [Kombinierte Klasse](kombiniert.md) | verglichen | Random Forest, RBF-SVM, LSTM | `combined_svm_aez_month` (BAcc kombiniert 0.869, 80/20) | Überarbeitetes GRU erreicht 0.832; Kontext ist entscheidend, die SVM bleibt klar führend. 10 Sentinel-2-Bänder halten die Qualität ([Datenreduktion](../datenreduktion.md)) |
| [Hierarchisch](hierarchisch.md) | verglichen | RBF-SVM (bester), Extra Trees, HistGradientBoosting, MLP, logistische Regression, kNN, PLS-DA, LSTM | `hierarchical_svm` (BAcc kombiniert 0.844) | LSTM-Embeddings erreichen 0.568; hinter der SVM. Bei unbekannten Aufnahmen robuster als kombiniert ([Datenreduktion](../datenreduktion.md#grenzen-unbekannte-aufnahmen)) |

Status: *offen* → *geplant* → *in Arbeit* → *verglichen* → *gewählt* / *verworfen*.

!!! note "Split-Protokolle"
    Die Werte stammen aus zwei Protokollen (bis 2026-10-20 70/30, danach 80/20) und sind nur
    innerhalb desselben Protokolls vergleichbar; Details im [Experimentlog](../experimente.md).

## Ansatz × Klassifikator

Beste BAcc kombiniert auf der Validierung je Kombination – ein Klick auf die Zahl führt zum
Unterkapitel. Eine neue Zeile, sobald ein Klassifikator in irgendeinem Ansatz getestet wurde.

| Klassifikator | Baseline | Getrennt | Kombiniert | Hierarchisch | Multi-Task |
| --- | --- | --- | --- | --- | --- |
| Dummy | [0.044](baseline.md#dummy) | | | | |
| Random Forest | [0.743](baseline.md#random-forest) | | [0.779](kombiniert.md#random-forest) | | |
| Multi-Task-MLP | | | | | [0.765](multi-task.md#tabellarisches-multi-task-mlp) |
| LSTM/GRU | | | [0.832](../datenreduktion.md#sequenzmodelle) | [0.568](hierarchisch.md#lstm-lstm-mit-hartem-routing) | [0.417](multi-task.md#multi-task-lstm-mit-spat-fusioniertem-kontext) |
| RBF-SVM | | | [**0.869**](../experimente.md) | [0.844](hierarchisch.md#rbf-svm-rbf-svm) | |
| Extra Trees | | | [0.804](kombiniert.md#extra-trees) | [0.785](hierarchisch.md#extra-trees-extra-trees) | |
| Extra Trees → RBF-SVM | | | | [0.801](hierarchisch.md#extra-trees-rbf-svm) | |
| LDA (gleiche Priors) | | | | [0.676](hierarchisch.md#lda-lda) | |
| Logistische Regression | | | | [0.759](hierarchisch.md#logistische-regression-logistische-regression) | |
| PLS-DA | | | | [0.686](hierarchisch.md#pls-da-pls-da) | |
| kNN | | | | [0.704](hierarchisch.md#knn-knn) | |
| MLP | | | | [0.787](hierarchisch.md#mlp-mlp) | |
| HistGradientBoosting | | | [0.777](kombiniert.md#histgradientboosting) | [0.795](hierarchisch.md#histgradientboosting-histgradientboosting) | |

So sieht man auf einen Blick, ob ein Klassifikator in allen Ansätzen gut ist oder ob ein Ansatz
mit jedem Klassifikator besser abschneidet.
Quer zu allen Ansätzen: [Datenrepräsentation](datenrepraesentation.md) (tabellarisch oder
sequenziell).

## Aufbau einer Ansatz-Seite

```markdown
# <Ansatz>

## Idee
Was der Ansatz modelliert und warum er für unsere Aufgabe in Frage kommt.

## Vor- und Nachteile
| Vorteil | Nachteil |

## Klassifikatoren

### <Klassifikator, z. B. Random Forest – bei zwei Stufen: Random Forest → SVM>
- Einstellungen (per `tune()` gefunden), Vorverarbeitung
- Ergebnis: BAcc Crop / Stage / kombiniert, CV-Score – Notebook und MLflow-Run verlinken
- Erkenntnisse (2–5 Stichpunkte)

## Fazit
Status, Entscheidung, Begründung.
```

Wie ein Ansatz entsteht und bewertet wird: [Einen Ansatz entwickeln](../ansatz-entwickeln.md).
