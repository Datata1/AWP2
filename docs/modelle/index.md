# Modelle

Welche Modellierungsansätze wir verfolgen und wie gut sie sind. Was Kulturarten und
Entwicklungsstadien fachlich bedeuten, steht in der [Domäne](../domaene/kulturpflanzen.md).

| Seite | Inhalt |
| --- | --- |
| [Einen Ansatz entwickeln](ansatz-entwickeln.md) | Der Ablauf: bauen → `tune()` → `run()` → festhalten → vergleichen |
| [Ansätze](ansaetze/index.md) | Welche Strategien wir verfolgen – mit Begründung, Status und Entscheidung |
| [Klassifikatoren](klassifikatoren.md) | Kandidaten nach Modellfamilie, Voraussetzungen und Priorität |
| [Experimente](experimente.md) | Log aller Läufe mit Scores |
| [Fehleranalyse](fehleranalyse.md) | Wo und warum die Ansätze falsch liegen – Vergleich Baseline, kombiniert, hierarchisch |
| [Datenreduktion](datenreduktion.md) | Meilenstein 3: Qualität vs. Input, Sentinel-2-Empfehlung, Grenzen bei unbekannten Aufnahmen |
| [Feature Importance](feature-importance.md) | Welche Wellenlängen und Metadaten die Referenz-SVM nutzt – Permutation Importance, SHAP und einzeln betrachtete Spektralgruppen |

## Aktueller Stand

| | |
| --- | --- |
| Bestes Modell | `combined_svm_aez_month` (Kombinierte Klasse, MLflow-Run `17dc9e16`, 80/20-Split) |
| BAcc Crop / Stage / kombiniert | 0.923 / 0.922 / **0.869** |
| Macro-F1 kombiniert · Samples-F1 | 0.859 · 0.920 |

Wichtigste Erkenntnis: Die erweiterte Suche verbessert die skalierte RBF-SVM im Ansatz mit
kombinierten Klassen deutlich und verhindert ungültige Crop/Stage-Paare. Fehler entstehen
weiterhin vor allem zwischen Winterweizen und Baumwolle, Mais und Soja sowie bei ähnlichen
Stadien. Nächster Schritt: eine getrennte Preprocessing-Hypothese für die neue SVM-Referenz
prüfen.
