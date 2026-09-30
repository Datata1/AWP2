# Experimente

Jedes Experiment bekommt eine Zeile (wird von `/experiment` automatisch ergänzt). Metriken aus
`run()` auf dem gemeinsamen Validierungs-Split; Details jedes Laufs lokal in MLflow unter der
Run-ID ([Experimente & MLflow](../entwicklung/tracking.md)).

| Datum | ID | Autor | Ansatz | BAcc Crop | BAcc Stage | BAcc kombiniert | Macro-F1 kombiniert | Samples-F1 | MLflow-Run | Notiz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-30 | dummy_most_frequent | Datata1 | Baseline | 0.200 | 0.167 | 0.044 | 0.009 | 0.326 | f8cf4ecc | Untergrenze: immer die häufigste Klasse |
| 2026-09-30 | rf_baseline | Datata1 | Baseline | 0.890 | 0.870 | 0.743 | 0.708 | 0.878 | 9641f6a8 | Random Forest, Crop+Stage gemeinsam; Tuning `rf_search` (2f90ad7e), CV 0.750 |
