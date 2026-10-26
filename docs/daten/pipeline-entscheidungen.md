# Pipeline: Warum so?

Die Entscheidungen hinter der [Pipeline](pipeline.md) – jeweils mit Grund, damit spätere
Änderungen bewusst passieren.

## Datenaufbereitung (`make data`)

| Entscheidung | Grund |
| --- | --- |
| Rohdaten beim Laden per pandera-Schema prüfen | Formatfehler (falsche Labels, Spalten, Werte) fallen sofort auf statt erst im Modell |
| 2 exakt doppelte Zeilen entfernen | Sonst kann dieselbe Messung in Train **und** Validierung landen – das Modell würde „auswendig“ richtig liegen |
| Konfigurierbarer, fixer Holdout (Standard: 80/20, bis 2026-10-20: 70/30) | `make data DATA_ARGS="--validation-size …"` erzeugt einen reproduzierbaren Split; Vergleiche gelten nur innerhalb desselben Protokolls. 80/20 gibt den Modellen mehr Trainingszeilen (CV 0,848 → 0,859); alle 23 Crop-Stage-Paare bleiben in Validierung und jedem Fold vertreten |
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
| `AEZ` und Monat als Kontextmerkmale | Für tabellarische Modelle wird `AEZ` one-hot-kodiert und `Month` standardmäßig durch `Month_sin` und `Month_cos` mit Jahresperiode 12 ersetzt; `use_cyclic_month=False` behält die numerische Monatszahl. Bei `scale=True` werden beide zyklischen Merkmale innerhalb des Trainingsfolds skaliert. LSTMs vergleichen zusätzlich kategoriale AEZ- und Monatsembeddings: `use_categorical_context=True` bewahrt dazu die beiden Rohspalten. Das Spektrum bleibt eine Wellenlängenfolge; Kontext wird erst nach dem LSTM mit der spektralen Repräsentation verbunden und nicht an jedem Wellenlängenschritt wiederholt. Kontextmerkmale sind laut Aufgabe erlaubt, können aber Orts- oder Kalenderabhängigkeit abbilden; daher ihren Nutzen und die Monatskodierung vergleichen. |
| Skalierung optional | Nötig für SVM, logistische Regression, MLP; für Baum-Modelle überflüssig |
| Höchstens ein Reduktionsverfahren gleichzeitig | Sonst ist der Anteil der Verfahren am Score nicht trennbar; jedes Verfahren läuft gegen PCA |
| NMF/OSP auf rohen Spektren | NMF braucht Nicht-Negativität (Reflektanz, multiplikative Updates), OSP die physikalischen Endmember – Standardisierung oder SNV würden beides zerstören |
| DBN als gestapelte Bernoulli-RBMs aus sklearn | Volles DBN-Training (Fine-Tuning, CAE) braucht PyTorch/Keras; für den Methodenvergleich genügen RBM-Merkmale auf Min-Max-skalierten Bändern |
| Zentrierung als Alternative zur Standardisierung | Zentrierung normiert keine Varianzen (schonender als Standardisierung); LPP läuft zusätzlich auch auf rohen Bändern |
| Mean Albedo als Zusatzmerkmal (`use_mean_albedo`) | SNV entfernt die Helligkeit per Konstruktion; der Mittelwert über die interpolierten Bänder hält sie vor SNV als eigenes Merkmal fest |
| Albedo mit derselben Methode wie die Spektren skalieren | Unskaliert hat die Albedo eine Std von ≈ 5 (Reflektanz in %), die standardisierten Bänder von 1 – ein Merkmal dominiert dann abstandsbasierte Modelle (SVM, kNN, logistische Regression, MLP). Hierarchische RBF-SVM, CV: + Albedo unskaliert 0.833, skaliert 0.839 (Basis 0.839); mit SNV 0.816 → 0.832 ([#119](https://github.com/Datata1/AWP2/issues/119)). Für Bäume ohne Wirkung |
| Vegetationsindizes aus interpolierten Reflektanzen (eigener Zweig) | Der Katalog ordnet jede Zielwellenlänge dem nächsten nicht leeren Hyperion-Band innerhalb von 5 nm zu und berechnet die Formeln aus Reflektanzanteilen. Auf SNV-standardisierten Spektren (Mittelwert 0, Varianz 1) wären die Verhältnisse bedeutungslos; deshalb nutzt der Zweig die Rohreflektanzen und wird erst nach der Indexauswahl mit derselben konfigurierten Methode wie die Spektren skaliert. Beide Skaler werden pro Trainingsfold separat gefittet. |
| AEZ × Monat als eigene Indikatoren (`use_aez_month`) | Kulturfehler häufen sich in bestimmten Regionen zu bestimmten Zeiten ([Fehleranalyse](../modelle/fehleranalyse.md)). Ein Indikator je Kombination macht diesen Kontext direkt sichtbar und gibt ihm in Abstandsmodellen mehr Gewicht. Nur im Training beobachtete Kombinationen; unbekannte ergeben lauter Nullen. Hierarchische SVM: Kulturfehler im Mai (Training, out-of-fold) 25,4 % → 22,1 %, Gesamt-CV aber nur im Rauschen ([#114](https://github.com/Datata1/AWP2/issues/114)) – optional, nicht Standard |
| Optionen als pydantic-`PreprocessingConfig` | Tippfehler oder falsche Typen (`scale="yes"`) fallen sofort auf |
| Spektralregionen vorab auswählen (`spectral_regions`) | Nur für Analysen: Das Modell sieht nur die genannten Regionen (Grenzen wie [Domänen-Doku](../domaene/vegetation.md)), `()` ergibt ein reines Metadatenmodell. Die Auswahl erfolgt **vor** Interpolation, Skalierung und allen Spektralzweigen, damit ausgeschlossene Bänder weder Nachbarwerte noch Skalierung beeinflussen. Feste Wellenlängenintervalle statt datengetriebener Auswahl, daher kein Leakage. Standard ist `None` (alle Bänder); Ergebnisse siehe [Feature Importance](../modelle/feature-importance.md), Abschnitt 6 |
| Pipeline-Grafik generiert (`pipeline_options_diagram`) | Handgezeichnete Grafiken veralten; die CI prüft Stand und Vollständigkeit (alle baubaren Schritte) |

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
