# Modelle

Welche Modellierungsansätze wir verfolgen und wie gut sie sind. Was Kulturarten und
Entwicklungsstadien fachlich bedeuten, steht in der [Domäne](../domaene/kulturpflanzen.md).

| Seite | Inhalt |
| --- | --- |
| [Einen Ansatz entwickeln](ansatz-entwickeln.md) | Der Ablauf: bauen → `tune()` → `run()` → festhalten → vergleichen |
| [Ansätze](ansaetze/index.md) | Welche Strategien wir verfolgen – mit Begründung, Status und Entscheidung |
| [Experimente](experimente.md) | Log aller Läufe mit Scores |

## Aktueller Stand

| | |
| --- | --- |
| Bestes Modell | `rf_baseline` (Baseline, MLflow-Run `9641f6a8`) |
| BAcc Crop / Stage / kombiniert | 0.890 / 0.870 / **0.743** |
| Macro-F1 kombiniert · Samples-F1 | 0.708 · 0.878 |

Wichtigste Erkenntnis: Die Kultur ist gut trennbar, Fehler entstehen vor allem zwischen Mais und
Soja und zwischen benachbarten Stadien. Nächster Schritt (M2): begründete Ansätze gegen diese
Baseline vergleichen – siehe [Ansätze](ansaetze/index.md).
