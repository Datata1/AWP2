# Kombinierte Klasse

## Idee

Kultur und Stadium werden zu *einer* Klasse `Crop|Stage` verbunden – 23 Klassen im Datensatz.
Ein beliebiger Klassifikator lernt damit direkt die gemeinsame Verteilung P(Crop, Stage).
Baustein und Recherche: #54, Entwurf #55.

## Vor- und Nachteile

| Vorteil | Nachteil |
| --- | --- |
| Sagt nur Paare vorher, die es gibt | Nutzt nicht, dass das Stadium von der Kultur abhängt: P(Crop, Stage) = P(Crop) · P(Stage \| Crop) wird als 23 unabhängige Klassen gelernt |
| Funktioniert mit jedem Klassifikator | Kein geteiltes Wissen zwischen z. B. Mais/Late und Soja/Late |
| Ein Modell, einfach zu vergleichen | Wenige Daten je Klasse (Baumwolle/Harvest: 11 Zeilen) |
| | Jeder Kultur-Fehler ist automatisch auch ein Stadium-Fehler |

!!! todo "Recherche (#54, Grundlagen in #11)"
    - Wird der Ansatz in der Literatur für hierarchische Labels genutzt, und mit welchem Ergebnis?
    - Wie verhält er sich gegenüber hierarchischer Klassifikation bei so seltenen Klassen?
    - Quellen in [Quellen](../../domaene/quellen.md) eintragen.

## Klassifikatoren

### Random Forest

Notebook: `notebooks/02_duac1011_mlflow-test.ipynb`. Der Random Forest lernt 23 kombinierte
Klassen und gibt seine Vorhersagen anschließend wieder als `Crop` und `Stage` aus.

- 300 Bäume, ausgewogene Klassengewichte und Standardvorverarbeitung mit AEZ/Month.
- Die Suche `combined_rf_search` (MLflow `b2cefd2`) verglich acht Konfigurationen; am besten
    war `max_depth=None`, `max_features=0.3`, `min_samples_leaf=1` mit CV-BAcc kombiniert
    0.788.
- Der einmalige Validierungslauf `combined_rf` (MLflow `5b7e6cef`) erreicht BAcc 0.874 für
    Crop, 0.876 für Stage und **0.779 kombiniert**; damit übertrifft er die Baseline um 0.036.
- Alle vorhergesagten Crop/Stage-Paare sind gültig. Schwächster Recall bleibt Mature_Senesc
    (0.81); die häufigste Kulturverwechslung ist Soja → Mais (58 Fälle).

![Tuning der kombinierten Klassen](../img/combined_rf_tuning.png)

![Confusion Matrices](../img/combined_rf_confusion.png)

![Recall je Klasse](../img/combined_rf_recall.png)

![Häufigste Verwechslungen](../img/combined_rf_confusions.png)

### RBF-SVM

Notebook: `notebooks/02_duac1011_combined-svm.ipynb`. Die SVM lernt dieselben 23 kombinierten
Klassen wie der Random Forest; Spektren und Month werden dafür standardisiert.

- Die Suche `combined_svm_search` (MLflow `9b1c0917`) verglich neun Kombinationen aus `C` und
    `gamma`. Beste Einstellung: `C=10.0`, `gamma=0.01`, CV-BAcc kombiniert 0.834.
- Der einmalige Validierungslauf `combined_svm` (MLflow `b2776f68`) erreicht BAcc 0.914 für
    Crop, 0.890 für Stage und **0.845 kombiniert**. Das übertrifft den kombinierten Random Forest
    um 0.066.
- Alle vorhergesagten Crop/Stage-Paare sind gültig. Bei Crop bleibt Winterweizen mit Recall 0.83
    am schwächsten; bei Stage sind Emerge_VEarly (0.82) und Mature_Senesc (0.83) am schwächsten.
- Die häufigsten Verwechslungen sind Winterweizen → Baumwolle (37), Soja → Mais (36) und
    Critical → Mature_Senesc (23).

![Tuning der RBF-SVM](../img/combined_svm_tuning.png)

![Confusion Matrices](../img/combined_svm_confusion.png)

![Recall je Klasse](../img/combined_svm_recall.png)

![Häufigste Verwechslungen](../img/combined_svm_confusions.png)

## Fazit

**Verglichen – vorläufig führend.** Die RBF-SVM verbessert die kombinierte BAcc gegenüber dem
Random Forest deutlich und verhindert weiterhin ungültige Paare. Gegen hierarchische und
Multi-Task-Ansätze bleibt der kombinierte Ansatz weiter zu vergleichen.
