# Endmodell und Abgabe-CSV

`test.csv` ist der finale, ungelabelte Validierungsdatensatz ([offene Fragen](../projekt/offene-fragen.md)).
Für die Abgabe wird das Modell mit festen Hyperparametern auf **allen** Trainingsdaten
(`train.csv`, 5 589 Zeilen nach Entfernen exakter Duplikate) trainiert und sagt `test.csv` vorher.
Ein eigener Validierungsanteil bleibt nicht zurück; Tuning und Bewertung sind vorher abgeschlossen.

| Skript | Modell | Validierung (80/20) | MLflow-Run |
| --- | --- | --- | --- |
| `tools/predict_combined_svm_aez_month.py` (`make predict`) | **Offizielles bestes Modell:** kombinierte RBF-SVM, `C=300`, `gamma=0.003`, Standardisierung, AEZ × Monat | BAcc kombiniert 0,869 | `17dc9e16` |
| `tools/predict_automl_final.py` (`make predict-automl`) | AutoML-Gewinner: kombinierte RBF-SVM, `C=100`, `gamma=scale`, MinMax, Savitzky-Golay 11/3, Albedo, Indizes, AEZ × Monat, Interpolationsgrenze 10 nm | BAcc kombiniert 0,883 | `c99e3915` |

Beide Skripte sind unabhängig, nutzen aber `awp2.submission`. Die Konfiguration ist je Skript
fest eingetragen; die dokumentierten Validierungswerte wurden mit ihr reproduziert (0,8686 und 0,8832).

```bash
make predict          # reports/predictions/combined_svm_aez_month.csv
make predict-automl   # reports/predictions/automl_final.csv
uv run python tools/predict_combined_svm_aez_month.py --test-file <csv> --output <csv>
```

Die CSV ist `test.csv` (`id`, `AEZ`, `Month`, alle Bänder) mit den zwei zusätzlichen Spalten
`Crop` und `Stage`. Vorhersagen sind immer gültige
Kultur-Stadium-Kombinationen, weil die Modelle auf der kombinierten Klasse trainieren.
