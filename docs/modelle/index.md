# Modelle

Welche Modellierungsansätze wir verfolgen und wie gut sie sind. Was Kulturarten und
Entwicklungsstadien fachlich bedeuten, steht in der [Domäne](../domaene/kulturpflanzen.md).

| Seite | Inhalt |
| --- | --- |
| [Einen Ansatz entwickeln](ansatz-entwickeln.md) | Der Ablauf: bauen → `tune()` → `run()` → festhalten → vergleichen |
| [Ansätze](ansaetze/index.md) | Welche Strategien wir verfolgen – mit Begründung, Status und Entscheidung |
| [Klassifikatoren](klassifikatoren.md) | Kandidaten nach Modellfamilie, Voraussetzungen und Priorität |
| [Experimente](experimente.md) | Log aller Läufe mit Scores |
| [Fehleranalyse](fehleranalyse.md) | Eingefrorenes AutoML-Modell (80/20): Klassenrecall und Verwechslungen; historischer Vergleich separat |
| [Feature Importance](feature-importance.md) | Welche Wellenlängen und Metadaten die Referenz-SVM nutzt – Permutation Importance, SHAP und einzeln betrachtete Spektralgruppen |

## Aktueller Stand

| | |
| --- | --- |
| Aktuelle 80/20-Referenz | `automl_final`: kombinierte RBF-SVM, MLflow-Run `c99e3915` |
| BAcc Crop / Stage / kombiniert | 0.9325 / 0.9204 / **0.8832** |
| Macro-F1 Crop / Stage / kombiniert | 0.9170 / 0.9107 / 0.8576 |
| Samples-F1 · ungültige Paare | 0.9217 · 0 % |
| Auswahl auf Trainingsfolds | 384 Kandidaten; Gewinner-CV 0.86682 ± 0.01744, Tuning `a08557a2` |

Die [fokussierte AutoML-Suche](ansaetze/automl.md) wählt eine kombinierte RBF-SVM mit
MinMax-Skalierung, Glättung, Albedo, Vegetationsindizes und AEZ×Monat-Kontext. Die Auswahl
erfolgte ausschließlich per CV; der Gewinner wurde anschließend einmal validiert.
[Konfiguration und Ergebnisse](ansaetze/kombiniert.md#rbf-svm-aus-dem-fokussierten-automl-lauf).

Ältere Referenzen wie `combined_svm_expanded` (0.866) beruhen auf 70/30 und sind nicht
direkt vergleichbar. Das mehrfach betrachtete gemeinsame Holdout ist kein unabhängiger
Abschlusstest; die adaptive Suche kann CV-Scores optimistisch machen. Die Modellwahl ist
eingefroren. Die [beschreibende Fehleranalyse](fehleranalyse.md#finales-automl-modell-8020)
ist abgeschlossen: 998/1.118 vollständig richtige Paare, 77 Kulturfehler und 43 reine
Stadiumfehler; die schwächste Paarklasse ist `soybean|Emerge_VEarly` (Recall 0.5714, n=21).
Das gespeicherte Modell reproduziert alle Metriken ohne erneutes Training.
Die Befunde dienen der Abschlussdokumentation, nicht weiterem Holdout-Tuning.

Ein [eigenständiges Sicherungspaket](ansaetze/automl.md#sicherung-und-wiederverwendung)
liegt lokal unter `models/automl_final.zip`. Das Laden und Vorhersagen wurde nach dem
Entpacken in einer frischen Umgebung geprüft; ein zusätzliches externes Backup steht noch aus.
