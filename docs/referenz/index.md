# API-Referenz

Automatisch aus den Docstrings im Code erzeugt – sie ist deshalb immer auf dem Stand des
Codes. Wie die Bausteine zusammenspielen, erklären die Seiten unter [Daten](../daten/index.md).

| Modul | Inhalt |
| --- | --- |
| [`awp2.config`](config.md) | Pfade, Spaltennamen, Labels und alle Konstanten |
| [`awp2.data`](data.md) | Artefakte laden (`load_split`, `load_folds`), Datensatz, Split, Rohdaten |
| [`awp2.preprocessing`](preprocessing.md) | Vorverarbeitung als sklearn-Transformer, `PreprocessingConfig` |
| [`awp2.evaluation`](evaluation.md) | `evaluate` → `Metrics`, `scorer` für CV, Confusion Matrices |
| [`awp2.plots`](plots.md) | Gemeinsame Plots, z. B. Spektren |

Neue öffentliche Funktionen brauchen einen Docstring im Google-Stil (`Args:`, `Returns:`,
`Raises:`) – `make lint` prüft das.
