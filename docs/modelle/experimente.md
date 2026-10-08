# Experimente

Jedes Experiment bekommt eine Zeile (wird von `/experiment` automatisch ergänzt). Metriken aus
`run()` auf dem gemeinsamen Validierungs-Split; Details jedes Laufs lokal in MLflow unter der
Run-ID ([Experimente & MLflow](../entwicklung/tracking.md)).

| Datum | ID | Autor | Ansatz | BAcc Crop | BAcc Stage | BAcc kombiniert | Macro-F1 kombiniert | Samples-F1 | MLflow-Run | Notiz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-30 | dummy_most_frequent | Datata1 | Baseline | 0.200 | 0.167 | 0.044 | 0.009 | 0.326 | f8cf4ecc | Untergrenze: immer die häufigste Klasse |
| 2026-09-30 | rf_baseline | Datata1 | Baseline | 0.890 | 0.870 | 0.743 | 0.708 | 0.878 | 9641f6a8 | Random Forest, Crop+Stage gemeinsam; Tuning `rf_search` (2f90ad7e), CV 0.750 |
| 2026-10-02 | rf_eda_baseline | duac1011 | Baseline | 0.890 | 0.870 | 0.743 | 0.708 | 0.878 | 6c2a119c | EDA-Merkmalsablage: Kontext gewinnt CV 0.750 vor Indizes 0.590 und Spektren 0.570; Tuning `rf_context_search` (7834b168), 0.3 % ungültige Paare |
| 2026-10-04 | combined_rf | duac1011 | Kombinierte Klasse | 0.874 | 0.876 | 0.779 | 0.762 | 0.880 | 5b7e6cef | Random Forest auf 23 kombinierten Klassen; Tuning `combined_rf_search` (b2cefd2), CV 0.788, keine ungültigen Paare |
| 2026-10-05 | combined_svm | duac1011 | Kombinierte Klasse | 0.914 | 0.890 | 0.845 | 0.804 | 0.890 | b2776f68 | RBF-SVM auf 23 kombinierten Klassen, skaliert; Tuning `combined_svm_search` (9b1c0917), CV 0.834, keine ungültigen Paare |
| 2026-10-08 | combined_svm_no_meta | duac1011 | Kombinierte Klasse | 0.871 | 0.838 | 0.779 | 0.725 | 0.844 | daa233c2 | RBF-SVM ohne AEZ und Month; Tuning `combined_svm_no_meta_search` (0afa2b8e), CV 0.742, keine ungültigen Paare |
| 2026-10-08 | combined_svm_indices | duac1011 | Kombinierte Klasse | 0.914 | 0.886 | 0.842 | 0.803 | 0.890 | 2998aea0 | RBF-SVM mit NDVI, NDRE, PRI und NDWI; Tuning `combined_svm_indices_search` (43fc0e63), CV 0.835, keine ungültigen Paare |
| 2026-10-08 | combined_cnn | duac1011 | Kombinierte Klasse | 0.819 | 0.712 | 0.626 | 0.559 | 0.741 | 1bd4f598 | Kleine 1D-CNN auf Spektralfolge mit AEZ/Month-Kontext; Tuning `combined_cnn_search` (d1cf7193), CV 0.611, keine ungültigen Paare |
| 2026-10-08 | combined_extra_trees | duac1011 | Kombinierte Klasse | 0.886 | 0.895 | 0.804 | 0.796 | 0.897 | 78877676 | Extra Trees auf 23 kombinierten Klassen; Tuning `combined_extra_trees_search` (915736b9), CV 0.812, keine ungültigen Paare |
| 2026-10-08 | combined_hist_gradient_boosting | duac1011 | Kombinierte Klasse | 0.885 | 0.881 | 0.777 | 0.773 | 0.891 | 6858b266 | HistGradientBoosting auf 23 kombinierten Klassen mit gewichteten Samples; Tuning `combined_hist_gradient_boosting_search` (b26e8cc8), CV 0.781, keine ungültigen Paare |
