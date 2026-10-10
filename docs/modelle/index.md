# Modelle

Welche Modellierungsansätze wir verfolgen und wie gut sie sind. Was Kulturarten und
Entwicklungsstadien fachlich bedeuten, steht in der [Domäne](../domaene/kulturpflanzen.md).

| Seite | Inhalt |
| --- | --- |
| [Einen Ansatz entwickeln](ansatz-entwickeln.md) | Der Ablauf: bauen → `tune()` → `run()` → festhalten → vergleichen |
| [Ansätze](ansaetze/index.md) | Welche Strategien wir verfolgen – mit Begründung, Status und Entscheidung |
| [Klassifikatoren](klassifikatoren.md) | Kandidaten nach Modellfamilie, Voraussetzungen und Priorität |
| [Experimente](experimente.md) | Log aller Läufe mit Scores |

## Aktueller Stand

| | |
| --- | --- |
| Bestes Modell | `combined_svm_expanded` (Kombinierte Klasse, MLflow-Run `65232eab`) |
| BAcc Crop / Stage / kombiniert | 0.931 / 0.906 / **0.866** |
| Macro-F1 kombiniert · Samples-F1 | 0.830 · 0.909 |

Wichtigste Erkenntnis: Die erweiterte Suche verbessert die skalierte RBF-SVM im Ansatz mit
kombinierten Klassen deutlich und verhindert ungültige Crop/Stage-Paare. Fehler entstehen
weiterhin vor allem zwischen Winterweizen und Baumwolle, Mais und Soja sowie bei ähnlichen
Stadien. Nächster Schritt: eine getrennte Preprocessing-Hypothese für die neue SVM-Referenz
prüfen.
