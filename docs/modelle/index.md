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
| Bestes Modell | `combined_rf` (Kombinierte Klasse, MLflow-Run `5b7e6cef`) |
| BAcc Crop / Stage / kombiniert | 0.874 / 0.876 / **0.779** |
| Macro-F1 kombiniert · Samples-F1 | 0.762 · 0.880 |

Wichtigste Erkenntnis: Kombinierte Klassen verhindern ungültige Crop/Stage-Paare und verbessern
die kombinierte BAcc gegenüber der Baseline. Fehler entstehen weiterhin vor allem zwischen Mais
und Soja sowie zwischen benachbarten Stadien. Nächster Schritt (M2): den Ansatz gegen
[hierarchische](ansaetze/hierarchisch.md) und Multi-Task-Modelle vergleichen.
