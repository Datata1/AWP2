# Pipeline: Warum so?

Die Entscheidungen hinter der [Pipeline](pipeline.md) – jeweils mit Grund, damit spätere
Änderungen bewusst passieren.

## Datenaufbereitung (`make data`)

| Entscheidung | Grund |
| --- | --- |
| Rohdaten beim Laden per pandera-Schema prüfen | Formatfehler (falsche Labels, Spalten, Werte) fallen sofort auf statt erst im Modell |
| 2 exakt doppelte Zeilen entfernen | Sonst kann dieselbe Messung in Train **und** Validierung landen – das Modell würde „auswendig“ richtig liegen |
| Fixer 70/30-Holdout statt freier Wahl | Vorgabe für M1; alle vergleichen auf denselben Daten |
| Split und Folds als Dateien in `data/` | Alle arbeiten nachweislich auf demselben Split; die Zuordnung ist ansehbar |
| Vorverarbeitung **nicht** speichern | Sie wird je Modell nur auf dem Train-Teil gefittet – gespeicherte Features würden Informationen aus der Validierung enthalten |

### Warum stratifiziert?

Ein zufälliger Split kann seltene Klassen ungleich verteilen: Von den 11 Zeilen `cotton|Harvest`
könnten zufällig alle im Training landen – dann lässt sich diese Klasse gar nicht bewerten, und
die Balanced Accuracy schwankt je nach Zufall stark. **Stratifiziert** heißt: Der Split behält den
Anteil jeder Klasse in beiden Teilen bei (hier je Crop/Stage-Paar). Weil `train_test_split` und
`StratifiedKFold` dafür nur *eine* Klasse pro Zeile akzeptieren, bilden wir den Schlüssel
`Crop|Stage`. Die Anteile in Train und Validierung weichen dadurch um höchstens 0,05
Prozentpunkte voneinander ab.

### Warum der Holdout nicht zum Tunen dient

Wer Hyperparameter so lange ändert, bis der Holdout-Score gut ist, tuned auf den
Validierungsdaten – der Score wird zu optimistisch. Deshalb gibt es fünf CV-Folds im Train-Teil
fürs Tuning; der Holdout bleibt für den Endvergleich.

## Vorverarbeitung

| Schritt | Grund |
| --- | --- |
| 67 komplett leere Bänder entfernen → 131 Bänder | Keine Information (Wasserabsorption/Randbänder, siehe [EDA](eda.md)) |
| Einzelne Lücken spektral auffüllen | Linear nach Wellenlänge interpolieren, wenn die beiden gemessenen Stützbänder höchstens 15 nm auseinanderliegen; sonst den näheren Stützwert übernehmen. Am Spektralrand ebenfalls den nächsten Messwert verwenden. Vollständig leere Spektren werden mit den im Training bestimmten Bandmedianen gefüllt. Die Grenze ist über `PreprocessingConfig(max_interpolation_gap_nm=...)` konfigurierbar. Nach der Duplikatbereinigung betrifft die Imputation 43 Trainings- und 11 Testzeilen; in den Rohdaten sind es 44 Trainingszeilen. |
| `AEZ` und Monat als Kontextmerkmale | `AEZ` wird one-hot-kodiert. `Month` wird standardmäßig durch `Month_sin` und `Month_cos` mit Jahresperiode 12 ersetzt; `use_cyclic_month=False` behält die numerische Monatszahl. Bei `scale=True` werden beide zyklischen Merkmale innerhalb des Trainingsfolds skaliert. Kontextmerkmale sind laut Aufgabe erlaubt, können aber Orts- oder Kalenderabhängigkeit abbilden; daher ihren Nutzen und die Monatskodierung vergleichen. |
| Skalierung optional | Nötig für SVM, logistische Regression, MLP; für Baum-Modelle überflüssig |
| Optionen als pydantic-`PreprocessingConfig` | Tippfehler oder falsche Typen (`scale="yes"`) fallen sofort auf |

## Klassenungleichgewicht

Reis hat 93 Zeilen, Mais 2097. Modelle mit `class_weight` bekommen `class_weight="balanced"`,
alle anderen `balanced_sample_weight` – beides gewichtet je Crop/Stage-Paar, passend zur
Balanced Accuracy als Hauptmetrik.

## Offen – kommt nach der EDA (#20, #30)

- Glättung der Spektren (z. B. Savitzky-Golay)
- Umgang mit Ausreißern
- Bandauswahl, Vegetationsindizes als Features
- Split-Strategie, falls räumliche Cluster gefunden werden (dann Gruppen in `cv_splits`)
- `Month` ist zyklisch; ggf. als sin/cos kodieren
