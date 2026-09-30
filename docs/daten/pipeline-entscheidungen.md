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
| Lücken entlang der Wellenlänge interpolieren | Nachbarbänder sind stark korreliert – das schätzt besser als ein Spaltenmittel. Am Rand zählt das nächste Band (Spektren sind unterschiedlich hell, ein globaler Median läge daneben). Betrifft 43 Zeilen train und **11 Zeilen test** – muss daher auch bei der Vorhersage greifen |
| `AEZ` und `Month` optional als Features | Laut Aufgabe erlaubt; abschaltbar, um ihren Nutzen zu messen |
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
- 2 Spektren in `test.csv` sind identisch mit Trainingszeilen – für die EDA (#15) notiert
- `Month` ist zyklisch; ggf. als sin/cos kodieren
