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
| Bestes Modell | `combined_svm` (Kombinierte Klasse, MLflow-Run `b2776f68`) |
| BAcc Crop / Stage / kombiniert | 0.914 / 0.890 / **0.845** |
| Macro-F1 kombiniert · Samples-F1 | 0.804 · 0.890 |

Wichtigste Erkenntnis: Die skalierte RBF-SVM verbessert die kombinierte BAcc im Ansatz mit
kombinierten Klassen deutlich und verhindert ungültige Crop/Stage-Paare. Fehler entstehen
weiterhin vor allem zwischen Winterweizen und Baumwolle, Mais und Soja sowie bei ähnlichen
Stadien. Nächster Schritt (M2): den Ansatz gegen [hierarchische](ansaetze/hierarchisch.md) und
Multi-Task-Modelle vergleichen.
